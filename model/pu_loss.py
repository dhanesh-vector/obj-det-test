import torch
import torch.nn as nn
from .loss import YOLOELoss

def bbox_iou(box1, box2, eps=1e-7):
    """
    Calculate IoU between two sets of bounding boxes
    box1: (N, 4) xyxy format
    box2: (N, 4) xyxy format
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + eps)
    return iou


class YOLOEPUFocalLoss(YOLOELoss):
    """
    Variant of YOLOE Loss combining Soft Sampling (Gradient Re-weighting) and Focal IoU Weighting.
    Designed specifically for PU (Positive-Unlabeled) scenarios where a large percentage 
    of objects might not be labeled (e.g. 10% labeling rate).
    """
    
    def __init__(self, num_classes=80, reg_max=16, strides=[32, 16, 8], gamma=2.0, beta=1.0, label_smooth=0.0):
        super(YOLOEPUFocalLoss, self).__init__(num_classes, reg_max, strides, label_smooth=label_smooth)
        # Gamma for Soft Sampling (down-weighting highly confident but unlabeled backgrounds)
        self.gamma = gamma
        # Beta for Focal IoU (up-weighting positive anchors with high localization IoU)
        self.beta = beta
        
    def forward(self, predictions, targets, anchor_points, stride_tensor):
        """
        Compute PU-aware losses with Soft Sampling and Focal IoU.
        """
        cls_scores, reg_distri = predictions
        device = cls_scores.device
        batch_size = cls_scores.shape[0]
        num_anchors = cls_scores.shape[1]
        
        # Initialize losses
        loss_cls = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(device)
            gt_labels = targets[i]['labels'].to(device)
            
            pred_cls = cls_scores[i].clamp(1e-7, 1 - 1e-7)
            pred_reg = reg_distri[i]
            
            # Predict boxes for the whole image to compute Objectness and Focal IoU
            with torch.no_grad():
                pred_boxes_all = self._decode_boxes(pred_reg, anchor_points, stride_tensor)
                # Objectness score approximation (max across class scores for each anchor)
                obj_score, _ = torch.max(pred_cls, dim=-1)
            
            # If no ground truths are present in this image
            if len(gt_boxes) == 0:
                target_cls = torch.zeros_like(pred_cls)
                base_bce = self.bce(pred_cls, target_cls)  # (num_anchors, num_classes)

                # Soft Sampling: soften penalty for unlabelled potential positives.
                # Use sum()/num_anchors (i.e. mean) here since there are no positives
                # to normalize against — this image contributes a flat background term.
                soft_weight = (1.0 - obj_score).pow(self.gamma).unsqueeze(-1)
                loss_cls += (base_bce * soft_weight).sum() / num_anchors
                continue
            
            target_cls, target_box, fg_mask = self._assign_targets(gt_boxes, gt_labels, anchor_points, num_anchors, device)
            num_pos = fg_mask.sum().item()
            
            # Base classification loss computation
            base_cls_loss = self.bce(pred_cls, target_cls) # (num_anchors, num_classes)
            
            # Initialize weights for combination
            weight = torch.ones_like(obj_score) # (num_anchors)
            
            # 1. Soft Sampling Weight (Applied to Negative Anchors)
            # lambda = (1 - P_{objectness})^\gamma
            neg_mask = ~fg_mask
            weight[neg_mask] = (1.0 - obj_score[neg_mask]).pow(self.gamma)
            
            # 2. Focal IoU Weight (Applied to Positive Anchors)
            # lambda = (IoU_{pred, gt})^\beta
            if num_pos > 0:
                with torch.no_grad():
                    # Calculate IoU between predicted boxes and assigned GTs
                    pred_pos_boxes = pred_boxes_all[fg_mask]
                    tgt_pos_boxes = target_box[fg_mask]
                    iou = bbox_iou(pred_pos_boxes, tgt_pos_boxes)
                
                weight[fg_mask] = iou.pow(self.beta)
                
                # Normalize by num_pos (same as base loss fix): Focal IoU weights
                # modulate per-anchor contribution but total is still / num_pos.
                pos_loss = (base_cls_loss[fg_mask] * weight[fg_mask].unsqueeze(-1)).sum() / num_pos
            else:
                pos_loss = torch.tensor(0.0, device=device)

            # Soft Sampling negatives normalized by num_pos so neg total is
            # commensurate with pos total, not diluted across thousands of anchors.
            neg_loss = (base_cls_loss[neg_mask] * weight[neg_mask].unsqueeze(-1)).sum() / max(num_pos, 1)
            loss_cls += pos_loss + neg_loss
            
            # Box Regression Loss
            if fg_mask.any():
                pred_boxes = self._decode_boxes(pred_reg[fg_mask], anchor_points[fg_mask], stride_tensor[fg_mask])
                loss_box += nn.functional.smooth_l1_loss(pred_boxes, target_box[fg_mask], reduction='mean')
                
                # Compute DFL targets: (l, t, r, b) scaled by stride
                stride = stride_tensor[fg_mask].squeeze(-1)
                anchors = anchor_points[fg_mask]
                tgts = target_box[fg_mask]
                
                target_ltrb = torch.zeros_like(tgts)
                target_ltrb[:, 0] = (anchors[:, 0] - tgts[:, 0]) / stride
                target_ltrb[:, 1] = (anchors[:, 1] - tgts[:, 1]) / stride
                target_ltrb[:, 2] = (tgts[:, 2] - anchors[:, 0]) / stride
                target_ltrb[:, 3] = (tgts[:, 3] - anchors[:, 1]) / stride
                
                # Clamp to [0, reg_max - 0.01] to prevent out of bounds
                target_ltrb = target_ltrb.clamp(0, self.reg_max - 0.01)
                
                pred_distri = pred_reg[fg_mask].reshape(-1, self.reg_max + 1)
                target_ltrb = target_ltrb.reshape(-1)
                
                loss_dfl += self.dfl(pred_distri, target_ltrb).mean()
        
        loss_cls = loss_cls / batch_size * self.cls_weight
        loss_box = loss_box / batch_size * self.box_weight
        loss_dfl = loss_dfl / batch_size * self.dfl_weight
        
        return {
            'loss': loss_cls + loss_box + loss_dfl,
            'loss_cls': loss_cls,
            'loss_box': loss_box,
            'loss_dfl': loss_dfl
        }
