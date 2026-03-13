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
    
    def __init__(self, num_classes=80, reg_max=16, strides=[32, 16, 8], gamma=2.0, beta=1.0):
        super(YOLOEPUFocalLoss, self).__init__(num_classes, reg_max, strides)
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
                base_bce = self.bce(pred_cls, target_cls) # (num_anchors, num_classes)
                
                # Soft Sampling: lambda_soft_sampling = (1 - P_{objectness})^\gamma
                # Soften penalty for false positives because they might be unlabelled real objects
                soft_weight = (1.0 - obj_score).pow(self.gamma).unsqueeze(-1)
                loss_cls += (base_bce * soft_weight).mean()
                continue
            
            num_gts = len(gt_boxes)
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
            distances = torch.cdist(anchor_points, gt_centers)
            
            target_cls = torch.zeros_like(pred_cls)
            target_box = torch.zeros((num_anchors, 4), device=device)
            fg_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
            
            # Multi-anchor assignment per GT
            k = min(13, num_anchors // max(num_gts, 1))
            k = max(k, 3)
            
            for gt_idx in range(num_gts):
                _, topk_indices = torch.topk(distances[:, gt_idx], k=k, largest=False)
                gt_box = gt_boxes[gt_idx]
                
                # Check anchors inside GT box
                inside = (
                    (anchor_points[topk_indices, 0] >= gt_box[0]) &
                    (anchor_points[topk_indices, 0] <= gt_box[2]) &
                    (anchor_points[topk_indices, 1] >= gt_box[1]) &
                    (anchor_points[topk_indices, 1] <= gt_box[3])
                )
                
                assigned = topk_indices[inside] if inside.any() else topk_indices[:3]
                
                for idx in assigned:
                    target_cls[idx, gt_labels[gt_idx]] = 1.0
                    target_box[idx] = gt_box
                    fg_mask[idx] = True
            
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
                
                # Apply class balance scaling and Focal IoU weight
                pos_weight = min(num_anchors / num_pos, 50.0)
                pos_loss = (base_cls_loss[fg_mask] * weight[fg_mask].unsqueeze(-1)).mean() * pos_weight
            else:
                pos_loss = torch.tensor(0.0, device=device)
            
            # Compute Negative loss with Soft Sampling weight
            neg_loss = (base_cls_loss[neg_mask] * weight[neg_mask].unsqueeze(-1)).mean()
            loss_cls += pos_loss + neg_loss
            
            # Box Regression Loss
            if fg_mask.any():
                pred_boxes = self._decode_boxes(pred_reg[fg_mask], anchor_points[fg_mask], stride_tensor[fg_mask])
                loss_box += nn.functional.smooth_l1_loss(pred_boxes, target_box[fg_mask], reduction='mean')
        
        loss_cls = loss_cls / batch_size * self.cls_weight
        loss_box = loss_box / batch_size * self.box_weight
        
        return {
            'loss': loss_cls + loss_box + loss_dfl * self.dfl_weight,
            'loss_cls': loss_cls,
            'loss_box': loss_box,
            'loss_dfl': loss_dfl
        }
