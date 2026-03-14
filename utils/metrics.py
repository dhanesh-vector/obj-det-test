import torch
from torchvision.ops import nms, box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def decode_predictions(cls_scores, reg_dists, anchor_points, stride_tensor, conf_thresh=0.05, iou_thresh=0.5):
    """
    Decode raw model outputs to bounding boxes and apply NMS.
    """
    b, num_classes, num_anchors = cls_scores.shape
    
    scores, labels = cls_scores.max(dim=1) # (b, num_anchors)
    
    reg_dists = reg_dists.permute(0, 2, 1) # (b, num_anchors, 4)
    x1 = anchor_points[:, 0] - reg_dists[:, :, 0] * stride_tensor.squeeze()
    y1 = anchor_points[:, 1] - reg_dists[:, :, 1] * stride_tensor.squeeze()
    x2 = anchor_points[:, 0] + reg_dists[:, :, 2] * stride_tensor.squeeze()
    y2 = anchor_points[:, 1] + reg_dists[:, :, 3] * stride_tensor.squeeze()
    
    boxes = torch.stack([x1, y1, x2, y2], dim=-1) # (b, num_anchors, 4)
    
    batch_preds = []
    for i in range(b):
        mask = scores[i] > conf_thresh
        b_boxes = boxes[i][mask]
        b_scores = scores[i][mask]
        b_labels = labels[i][mask]
        
        if len(b_boxes) > 0:
            keep = nms(b_boxes, b_scores, iou_thresh)
            b_boxes = b_boxes[keep]
            b_scores = b_scores[keep]
            b_labels = b_labels[keep]
            
        batch_preds.append({
            'boxes': b_boxes,
            'scores': b_scores,
            'labels': b_labels
        })
        
    return batch_preds

class Evaluator:
    def __init__(self):
        self.metric = MeanAveragePrecision(iou_type="bbox")
        self.metric.warn_on_many_detections = False
        self._all_preds = []
        self._all_targets = []

    def update(self, batch_preds, device_targets):
        self.metric.update(batch_preds, device_targets)
        self._all_preds.extend(batch_preds)
        self._all_targets.extend(device_targets)

    def compute(self):
        """
        Compute mAP metrics plus precision and recall at IoU=0.5, conf=0.5.
        Returns a dictionary of regular floats.
        """
        mAP_results = self.metric.compute()
        parsed_results = {k: v.item() for k, v in mAP_results.items()}
        precision, recall = self._precision_recall_at_threshold(
            self._all_preds, self._all_targets, iou_thresh=0.5, conf_thresh=0.5
        )
        parsed_results['precision'] = precision
        parsed_results['recall'] = recall
        return parsed_results

    def _precision_recall_at_threshold(self, preds, targets, iou_thresh=0.5, conf_thresh=0.5):
        tp = fp = fn = 0
        for pred, target in zip(preds, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            gt_boxes = target['boxes']

            # Filter predictions by confidence
            mask = pred_scores >= conf_thresh
            pred_boxes = pred_boxes[mask]

            n_pred = len(pred_boxes)
            n_gt = len(gt_boxes)

            if n_gt == 0 and n_pred == 0:
                continue
            elif n_gt == 0:
                fp += n_pred
                continue
            elif n_pred == 0:
                fn += n_gt
                continue

            # Greedy IoU matching (highest IoU first)
            iou = box_iou(pred_boxes, gt_boxes)  # (n_pred, n_gt)
            matched_preds = set()
            matched_gts = set()
            for flat_idx in iou.flatten().argsort(descending=True):
                pi = (flat_idx // n_gt).item()
                gi = (flat_idx % n_gt).item()
                if iou[pi, gi] < iou_thresh:
                    break
                if pi not in matched_preds and gi not in matched_gts:
                    matched_preds.add(pi)
                    matched_gts.add(gi)

            tp += len(matched_preds)
            fp += n_pred - len(matched_preds)
            fn += n_gt - len(matched_gts)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall

    def reset(self):
        self.metric.reset()
        self._all_preds.clear()
        self._all_targets.clear()
