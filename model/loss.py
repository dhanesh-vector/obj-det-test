import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# P3 — CIoU loss helper
# ---------------------------------------------------------------------------

def ciou_loss(pred_boxes, target_boxes, eps=1e-9):
    """
    Complete IoU loss (Zheng et al. 2020) for (N, 4) xyxy-format boxes.

    Improves over smooth-L1 by jointly penalising:
      - overlap (IoU term)
      - centre-point distance normalised by enclosing diagonal (ρ²/c²)
      - aspect-ratio consistency (αv)

    Returns scalar mean loss over N positive anchors.
    Handles degenerate predicted boxes (negative w/h) via .clamp(min=0).
    """
    pw = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0)
    ph = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    gw = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0)
    gh = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)

    pcx = (pred_boxes[:, 0] + pred_boxes[:, 2]) * 0.5
    pcy = (pred_boxes[:, 1] + pred_boxes[:, 3]) * 0.5
    gcx = (target_boxes[:, 0] + target_boxes[:, 2]) * 0.5
    gcy = (target_boxes[:, 1] + target_boxes[:, 3]) * 0.5

    # Intersection
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    union = pw * ph + gw * gh - inter + eps
    iou = inter / union

    # Enclosing box diagonal squared (c²)
    c_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    c_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    c_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    c_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    c2 = (c_x2 - c_x1).clamp(0).pow(2) + (c_y2 - c_y1).clamp(0).pow(2) + eps

    # Centre-point distance squared (ρ²)
    rho2 = (pcx - gcx).pow(2) + (pcy - gcy).pow(2)

    # Aspect-ratio term (v, α) — no gradient through α per the paper
    with torch.no_grad():
        v = (4.0 / (math.pi ** 2)) * (
            torch.atan(gw / (gh + eps)) - torch.atan(pw / (ph + eps))
        ).pow(2)
        alpha_v = v / (1.0 - iou + v + eps)

    return (1.0 - iou + rho2 / c2 + alpha_v * v).mean()


# ---------------------------------------------------------------------------
# Distribution Focal Loss (unchanged)
# ---------------------------------------------------------------------------

class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL)
    https://ieeexplore.ieee.org/document/9792391
    """
    def __init__(self):
        super(DistributionFocalLoss, self).__init__()

    def forward(self, pred_dist, target):
        """
        pred_dist: (N, reg_max + 1) predicted distribution
        target: (N,) continuous target value
        """
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left

        loss = (
            F.cross_entropy(pred_dist, target_left, reduction='none') * weight_left +
            F.cross_entropy(pred_dist, target_right, reduction='none') * weight_right
        )
        return loss


# ---------------------------------------------------------------------------
# Main loss class
# ---------------------------------------------------------------------------

class YOLOELoss(nn.Module):
    """
    YOLOE loss with:
      P3 — CIoU box regression instead of smooth-L1
      P4 — Task-Aligned Assignment (TAL, TOOD 2021) instead of distance top-k
      P5 — Online Hard Example Mining (OHEM) for negatives
    """

    def __init__(self, num_classes=80, reg_max=16, strides=[32, 16, 8],
                 label_smooth=0.0,
                 tal_topk=13, tal_alpha=0.5, tal_beta=6.0,
                 ohem_ratio=3):
        super(YOLOELoss, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.label_smooth = label_smooth

        # P4 — TAL hyperparameters (TOOD paper defaults)
        self.tal_topk = tal_topk      # candidates per GT
        self.tal_alpha = tal_alpha    # cls-score exponent in alignment metric
        self.tal_beta = tal_beta      # IoU exponent in alignment metric

        # P5 — OHEM: keep top (ohem_ratio × num_pos) hardest negatives per image
        self.ohem_ratio = ohem_ratio

        # Loss weights
        self.cls_weight = 1.0
        self.box_weight = 2.5
        self.dfl_weight = 0.5

        self.bce = nn.BCELoss(reduction='none')
        self.dfl = DistributionFocalLoss()

    # -----------------------------------------------------------------------
    # P4 — Task-Aligned Assignment
    # -----------------------------------------------------------------------

    def _assign_targets(self, gt_boxes, gt_labels, anchor_points, num_anchors, device,
                        pred_cls=None, pred_boxes=None):
        """
        Task-Aligned Assignment (TAL).

        For each GT, selects the top-k anchor candidates whose alignment metric
            t = cls_score^tal_alpha × IoU(pred_box, gt_box)^tal_beta
        is highest, subject to the anchor being geometrically inside the GT box.
        Conflicts (anchor matched to multiple GTs) are resolved by keeping the GT
        with the highest IoU.

        Falls back to distance-based top-k if pred_cls / pred_boxes are absent
        (e.g., the very first forward pass before any predictions exist).
        """
        num_gts = len(gt_boxes)
        target_cls = torch.zeros((num_anchors, self.num_classes), device=device)
        target_box = torch.zeros((num_anchors, 4), device=device)
        fg_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)

        if num_gts == 0:
            return target_cls, target_box, fg_mask

        if pred_cls is None or pred_boxes is None:
            return self._assign_targets_distance(
                gt_boxes, gt_labels, anchor_points, num_anchors, device,
                target_cls, target_box, fg_mask)

        # ------------------------------------------------------------------
        # TAL core
        # ------------------------------------------------------------------
        A, G = num_anchors, num_gts

        # 1. IoU between every (anchor predicted box, GT box) pair → (A, G)
        inter_x1 = torch.max(pred_boxes[:, 0:1], gt_boxes[:, 0].unsqueeze(0))
        inter_y1 = torch.max(pred_boxes[:, 1:2], gt_boxes[:, 1].unsqueeze(0))
        inter_x2 = torch.min(pred_boxes[:, 2:3], gt_boxes[:, 2].unsqueeze(0))
        inter_y2 = torch.min(pred_boxes[:, 3:4], gt_boxes[:, 3].unsqueeze(0))
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)  # (A, G)

        pred_area = ((pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(0) *
                     (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(0)).unsqueeze(1)  # (A, 1)
        gt_area = ((gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(0) *
                   (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(0)).unsqueeze(0)        # (1, G)
        iou = inter / (pred_area + gt_area - inter + 1e-9)  # (A, G)

        # 2. Cls score for each (anchor, GT) pair
        # pred_cls: (A, C).  Index column = class of each GT → (A, G)
        cls_score = pred_cls[:, gt_labels].clamp(1e-7, 1.0 - 1e-7)  # (A, G)

        # 3. Alignment metric
        align = cls_score.pow(self.tal_alpha) * iou.pow(self.tal_beta)  # (A, G)

        # 4. Restrict to anchors inside the GT box
        ap_x = anchor_points[:, 0:1]   # (A, 1)
        ap_y = anchor_points[:, 1:2]
        in_gt = ((ap_x > gt_boxes[:, 0]) & (ap_x < gt_boxes[:, 2]) &
                 (ap_y > gt_boxes[:, 1]) & (ap_y < gt_boxes[:, 3]))  # (A, G)

        # Floor: guarantee every in-box anchor has non-zero alignment even when
        # IoU^tal_beta underflows to zero at random initialisation (iou ≈ 0.003
        # for stride-16 anchors → iou^6 ≈ 1e-15, below the 1e-9 guard below).
        # The floor is 100x below the 1e-9 guard so it never influences the
        # top-k ranking once the model produces any meaningful predictions.
        align = align * in_gt.float() + in_gt.float() * 1e-11

        # 5. Top-k anchors per GT
        topk = min(self.tal_topk, A)
        # align.t(): (G, A) → topk along anchor dim
        topk_vals, topk_idx = align.t().topk(topk, dim=1, largest=True)  # (G, topk)

        is_in_topk = torch.zeros(G, A, device=device)
        # Mark entries where alignment metric > 1e-11 floor (i.e., anchor inside GT)
        is_in_topk.scatter_(1, topk_idx, (topk_vals > 1e-11).float())
        is_in_topk = is_in_topk.t()  # (A, G)

        # 6. Candidate = inside GT box AND in top-k
        candidate = is_in_topk * in_gt.float()  # (A, G)

        # 7. Conflict resolution: for each anchor assigned to multiple GTs,
        #    keep the GT with the highest IoU
        candidate_iou = iou * candidate           # (A, G) — zero for non-candidates
        _, assigned_gt_idx = candidate_iou.max(dim=1)  # (A,)
        fg_mask = candidate.sum(dim=1).bool()     # (A,)

        if fg_mask.any():
            fg_gt_idx = assigned_gt_idx[fg_mask]  # (num_pos,)
            target_cls[fg_mask, gt_labels[fg_gt_idx]] = 1.0 - self.label_smooth
            target_box[fg_mask] = gt_boxes[fg_gt_idx]

        return target_cls, target_box, fg_mask

    def _assign_targets_distance(self, gt_boxes, gt_labels, anchor_points, num_anchors,
                                  device, target_cls, target_box, fg_mask):
        """
        Fallback: original distance-to-GT-centre top-k assignment.
        Used only during the very first forward before predictions are meaningful.
        """
        num_gts = len(gt_boxes)
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        distances = torch.cdist(anchor_points, gt_centers)

        k = min(13, num_anchors // max(num_gts, 1))
        k = max(k, 3)

        for gt_idx in range(num_gts):
            _, topk_indices = torch.topk(distances[:, gt_idx], k=k, largest=False)
            gt_box = gt_boxes[gt_idx]

            inside = (
                (anchor_points[topk_indices, 0] >= gt_box[0]) &
                (anchor_points[topk_indices, 0] <= gt_box[2]) &
                (anchor_points[topk_indices, 1] >= gt_box[1]) &
                (anchor_points[topk_indices, 1] <= gt_box[3])
            )
            assigned = topk_indices[inside] if inside.any() else topk_indices[:3]
            for idx in assigned:
                target_cls[idx, gt_labels[gt_idx]] = 1.0 - self.label_smooth
                target_box[idx] = gt_box
                fg_mask[idx] = True

        return target_cls, target_box, fg_mask

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, predictions, targets, anchor_points, stride_tensor):
        """
        Compute losses: CIoU box regression (P3), TAL assignment (P4), OHEM (P5).
        """
        cls_scores, reg_distri = predictions
        device = cls_scores.device
        batch_size = cls_scores.shape[0]
        num_anchors = cls_scores.shape[1]

        loss_cls = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)

        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(device)
            gt_labels = targets[i]['labels'].to(device)

            pred_cls = cls_scores[i].clamp(1e-7, 1 - 1e-7)
            pred_reg = reg_distri[i]

            if len(gt_boxes) == 0:
                # No GTs: all anchors are true negatives, plain mean BCE
                loss_cls += self.bce(pred_cls, torch.zeros_like(pred_cls)).mean()
                continue

            # Decode all predicted boxes for TAL (no gradient needed here)
            with torch.no_grad():
                pred_boxes_all = self._decode_boxes(pred_reg, anchor_points, stride_tensor)

            # P4 — TAL assignment
            target_cls, target_box, fg_mask = self._assign_targets(
                gt_boxes, gt_labels, anchor_points, num_anchors, device,
                pred_cls=pred_cls, pred_boxes=pred_boxes_all)

            num_pos = fg_mask.sum().item()

            # --- Classification loss ---
            base_cls_loss = self.bce(pred_cls, target_cls)  # (A, C)

            if num_pos > 0:
                pos_loss = base_cls_loss[fg_mask].sum() / num_pos
            else:
                pos_loss = torch.tensor(0.0, device=device)

            # P5 — OHEM: keep top (ohem_ratio × num_pos) hardest negative anchors
            neg_losses = base_cls_loss[~fg_mask].sum(dim=-1)  # (num_neg,)
            num_hard_neg = min(neg_losses.numel(), self.ohem_ratio * max(num_pos, 1))
            if num_hard_neg > 0:
                hard_neg_loss, _ = neg_losses.topk(num_hard_neg)
                neg_loss = hard_neg_loss.sum() / max(num_pos, 1)
            else:
                neg_loss = torch.tensor(0.0, device=device)

            loss_cls += pos_loss + neg_loss

            # --- Box and DFL losses (positive anchors only) ---
            if fg_mask.any():
                # P3 — CIoU box regression
                pred_boxes_pos = self._decode_boxes(
                    pred_reg[fg_mask], anchor_points[fg_mask], stride_tensor[fg_mask])
                loss_box += ciou_loss(pred_boxes_pos, target_box[fg_mask])

                # DFL
                stride = stride_tensor[fg_mask].squeeze(-1)
                anchors = anchor_points[fg_mask]
                tgts = target_box[fg_mask]

                target_ltrb = torch.zeros_like(tgts)
                target_ltrb[:, 0] = (anchors[:, 0] - tgts[:, 0]) / stride
                target_ltrb[:, 1] = (anchors[:, 1] - tgts[:, 1]) / stride
                target_ltrb[:, 2] = (tgts[:, 2] - anchors[:, 0]) / stride
                target_ltrb[:, 3] = (tgts[:, 3] - anchors[:, 1]) / stride
                target_ltrb = target_ltrb.clamp(0, self.reg_max - 0.01)

                pred_distri = pred_reg[fg_mask].reshape(-1, self.reg_max + 1)
                loss_dfl += self.dfl(pred_distri, target_ltrb.reshape(-1)).mean()

        loss_cls = loss_cls / batch_size * self.cls_weight
        loss_box = loss_box / batch_size * self.box_weight
        loss_dfl = loss_dfl / batch_size * self.dfl_weight

        return {
            'loss': loss_cls + loss_box + loss_dfl,
            'loss_cls': loss_cls,
            'loss_box': loss_box,
            'loss_dfl': loss_dfl,
        }

    def _decode_boxes(self, reg_distri, anchor_points, stride_tensor):
        """Decode regression distribution to xyxy boxes."""
        reg_distri = reg_distri.reshape(-1, 4, self.reg_max + 1)
        reg_distri = nn.functional.softmax(reg_distri, dim=-1)

        proj = torch.linspace(0, self.reg_max, self.reg_max + 1, device=reg_distri.device)
        reg_dist = (reg_distri * proj).sum(dim=-1)  # (N, 4) ltrb

        x1 = anchor_points[:, 0] - reg_dist[:, 0] * stride_tensor.squeeze()
        y1 = anchor_points[:, 1] - reg_dist[:, 1] * stride_tensor.squeeze()
        x2 = anchor_points[:, 0] + reg_dist[:, 2] * stride_tensor.squeeze()
        y2 = anchor_points[:, 1] + reg_dist[:, 3] * stride_tensor.squeeze()

        return torch.stack([x1, y1, x2, y2], dim=-1)
