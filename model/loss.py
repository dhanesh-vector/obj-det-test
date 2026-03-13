import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOELoss(nn.Module):
    """
    YOLOE loss function
    Combines classification loss, box regression loss, and DFL loss
    """
    
    def __init__(self, num_classes=80, reg_max=16, strides=[32, 16, 8]):
        super(YOLOELoss, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        
        # Loss weights
        self.cls_weight = 1.0
        self.box_weight = 2.5
        self.dfl_weight = 0.5
        
        # BCE for classification (sigmoid already applied in model)
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, predictions, targets, anchor_points, stride_tensor):
        """
        Compute losses with improved positive/negative sample handling
        """
        cls_scores, reg_distri = predictions
        device = cls_scores.device
        batch_size = cls_scores.shape[0]
        num_anchors = cls_scores.shape[1]
        
        # Initialize losses
        loss_cls = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        
        num_pos_total = 0
        
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(device)
            gt_labels = targets[i]['labels'].to(device)
            
            pred_cls = cls_scores[i].clamp(1e-7, 1 - 1e-7)
            pred_reg = reg_distri[i]
            
            if len(gt_boxes) == 0:
                target_cls = torch.zeros_like(pred_cls)
                loss_cls += self.bce(pred_cls, target_cls).mean()
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
            num_pos_total += num_pos
            
            # Focal-like weighting for classification
            if num_pos > 0:
                pos_weight = min(num_anchors / num_pos, 50.0)
                pos_loss = self.bce(pred_cls[fg_mask], target_cls[fg_mask]).mean() * pos_weight
            else:
                pos_loss = torch.tensor(0.0, device=device)
            
            neg_loss = self.bce(pred_cls[~fg_mask], target_cls[~fg_mask]).mean()
            loss_cls += pos_loss + neg_loss
            
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
    
    def _decode_boxes(self, reg_distri, anchor_points, stride_tensor):
        """Decode regression distribution to boxes"""
        # Simplified decoding - in practice use DFL
        # reg_distri: (N, 4 * (reg_max + 1))
        # Reshape and take weighted sum
        reg_distri = reg_distri.reshape(-1, 4, self.reg_max + 1)
        reg_distri = nn.functional.softmax(reg_distri, dim=-1)
        
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1, device=reg_distri.device)
        reg_dist = (reg_distri * proj).sum(dim=-1)  # (N, 4) - left, top, right, bottom
        
        # Decode to xyxy
        x1 = anchor_points[:, 0] - reg_dist[:, 0] * stride_tensor.squeeze()
        y1 = anchor_points[:, 1] - reg_dist[:, 1] * stride_tensor.squeeze()
        x2 = anchor_points[:, 0] + reg_dist[:, 2] * stride_tensor.squeeze()
        y2 = anchor_points[:, 1] + reg_dist[:, 3] * stride_tensor.squeeze()
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
