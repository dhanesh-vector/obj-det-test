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
        # Separate metric instance to get mAP@35 without polluting standard mAP averaging
        self.metric_35 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.35])
        self.metric_35.warn_on_many_detections = False
        self._all_preds = []
        self._all_targets = []

    def update(self, batch_preds, device_targets):
        self.metric.update(batch_preds, device_targets)
        self.metric_35.update(batch_preds, device_targets)
        self._all_preds.extend(batch_preds)
        self._all_targets.extend(device_targets)

    def compute(self):
        """
        Compute mAP metrics plus precision, recall, F1, SliceAP@50/75, mAP@35.
        Returns a dictionary of regular floats.
        """
        mAP_results = self.metric.compute()
        parsed_results = {k: v.item() for k, v in mAP_results.items() if v.numel() == 1}

        # mAP@35 from dedicated metric instance (map == map_35 since only one threshold)
        map_35_results = self.metric_35.compute()
        parsed_results['map_35'] = map_35_results['map'].item()

        precision, recall = self._precision_recall_at_threshold(
            self._all_preds, self._all_targets, iou_thresh=0.5, conf_thresh=0.5
        )
        parsed_results['precision'] = precision
        parsed_results['recall'] = recall
        p, r = precision, recall
        parsed_results['f1'] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        parsed_results['slice_ap_50'] = self._per_slice_ap(
            self._all_preds, self._all_targets, iou_thresh=0.5
        )
        parsed_results['slice_ap_75'] = self._per_slice_ap(
            self._all_preds, self._all_targets, iou_thresh=0.75
        )
        p_at_r80, r_at_p80 = self._operating_points(
            self._all_preds, self._all_targets, iou_thresh=0.5,
            fixed_recall=0.8, fixed_precision=0.8
        )
        parsed_results['precision_at_recall80'] = p_at_r80
        parsed_results['recall_at_precision80'] = r_at_p80
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

    def _per_slice_ap(self, preds, targets, iou_thresh=0.5):
        """
        Image-level (per-slice) AP at the given IoU threshold.

        Each slice is treated as a single sample:
          - Score:   max prediction confidence in the slice (0.0 if no predictions)
          - Label:   positive if the slice has at least one GT box
          - Hit:     positive slice is a TP if any prediction matches any GT at IoU >= iou_thresh

        The PR curve is built by sweeping the score threshold from high to low,
        and AP is the area under that curve (using the standard 101-point interpolation).
        """
        slice_scores = []
        slice_is_positive = []
        slice_is_hit = []

        for pred, target in zip(preds, targets):
            gt_boxes = target['boxes']
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']

            is_positive = len(gt_boxes) > 0
            score = pred_scores.max().item() if len(pred_scores) > 0 else 0.0

            hit = False
            if is_positive and len(pred_boxes) > 0:
                iou = box_iou(pred_boxes, gt_boxes)  # (n_pred, n_gt)
                hit = bool((iou >= iou_thresh).any())

            slice_scores.append(score)
            slice_is_positive.append(is_positive)
            slice_is_hit.append(hit)

        # Sort slices by score descending
        order = sorted(range(len(slice_scores)), key=lambda i: slice_scores[i], reverse=True)

        n_pos = sum(slice_is_positive)
        if n_pos == 0:
            return 0.0

        tp_cum = 0
        fp_cum = 0
        precisions = []
        recalls = []

        for i in order:
            if slice_is_positive[i]:
                if slice_is_hit[i]:
                    tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0.0)
            recalls.append(tp_cum / n_pos)

        # 101-point interpolated AP
        ap = 0.0
        for t in [r / 100 for r in range(101)]:
            p_at_t = max((p for p, r in zip(precisions, recalls) if r >= t), default=0.0)
            ap += p_at_t / 101

        return ap

    def _pr_curve(self, preds, targets, iou_thresh=0.5):
        """
        Build a box-level precision-recall curve by sweeping confidence threshold.

        For each prediction (across all images), greedy IoU matching at the given
        threshold determines whether it is a TP or FP. Predictions are then sorted
        by score descending and cumulative precision/recall are accumulated.

        Returns (precisions, recalls, score_thresholds) as plain Python lists in
        score-descending order. score_thresholds[i] is the confidence score of the
        detection that produced the i-th operating point on the curve.
        """
        scored_detections = []  # list of (score, is_tp: bool)
        n_gt_total = 0

        for pred, target in zip(preds, targets):
            gt_boxes = target['boxes']
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']

            n_gt_total += len(gt_boxes)

            if len(pred_boxes) == 0:
                continue

            is_tp = [False] * len(pred_boxes)
            if len(gt_boxes) > 0:
                iou = box_iou(pred_boxes, gt_boxes)  # (n_pred, n_gt)
                matched_gts = set()
                # Greedy match in score-descending order so high-conf preds get priority
                for pi in pred_scores.argsort(descending=True).tolist():
                    best_iou_val, best_gi = iou[pi].max(0)
                    best_gi = best_gi.item()
                    if best_iou_val.item() >= iou_thresh and best_gi not in matched_gts:
                        is_tp[pi] = True
                        matched_gts.add(best_gi)

            for i, score in enumerate(pred_scores.tolist()):
                scored_detections.append((score, is_tp[i]))

        if not scored_detections or n_gt_total == 0:
            return [0.0], [0.0], [0.0]

        scored_detections.sort(key=lambda x: x[0], reverse=True)

        tp_cum = 0
        fp_cum = 0
        precisions = []
        recalls = []
        score_thresholds = []
        for score, tp_flag in scored_detections:
            if tp_flag:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / n_gt_total)
            score_thresholds.append(score)

        return precisions, recalls, score_thresholds

    def _operating_points(self, preds, targets, iou_thresh=0.5,
                          fixed_recall=0.8, fixed_precision=0.8):
        """
        Derive two operating-point metrics from the PR curve:
          - Precision @ Recall >= fixed_recall  (how precise when catching fixed_recall of GTs)
          - Recall   @ Precision >= fixed_precision  (how many GTs caught at fixed_precision)

        Uses the maximum value satisfying the constraint (envelope-style), so the result
        is stable w.r.t. PR curve jaggedness.
        """
        precisions, recalls, _ = self._pr_curve(preds, targets, iou_thresh=iou_thresh)
        p_at_r = max((p for p, r in zip(precisions, recalls) if r >= fixed_recall), default=0.0)
        r_at_p = max((r for p, r in zip(precisions, recalls) if p >= fixed_precision), default=0.0)
        return p_at_r, r_at_p

    # ------------------------------------------------------------------ #
    # Public data-extraction helpers for external plotting                 #
    # ------------------------------------------------------------------ #

    def pr_curve_data(self, iou_thresh=0.5):
        """
        Returns (precisions, recalls, score_thresholds) as numpy arrays.
        Each entry corresponds to one operating point on the PR curve
        (sorted by score descending = recall ascending along the curve).
        """
        import numpy as np
        p, r, s = self._pr_curve(self._all_preds, self._all_targets, iou_thresh=iou_thresh)
        return np.array(p), np.array(r), np.array(s)

    def score_distributions(self, iou_thresh=0.5):
        """
        Returns (tp_scores, fp_scores) lists of prediction confidence values
        classified by whether the detection is a true positive at iou_thresh.
        Useful for plotting score-distribution histograms.
        """
        tp_scores, fp_scores = [], []
        for pred, target in zip(self._all_preds, self._all_targets):
            gt_boxes = target['boxes']
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            if len(pred_boxes) == 0:
                continue
            is_tp = [False] * len(pred_boxes)
            if len(gt_boxes) > 0:
                iou = box_iou(pred_boxes, gt_boxes)
                matched_gts = set()
                for pi in pred_scores.argsort(descending=True).tolist():
                    best_iou_val, best_gi = iou[pi].max(0)
                    best_gi = best_gi.item()
                    if best_iou_val.item() >= iou_thresh and best_gi not in matched_gts:
                        is_tp[pi] = True
                        matched_gts.add(best_gi)
            for i, score in enumerate(pred_scores.tolist()):
                (tp_scores if is_tp[i] else fp_scores).append(score)
        return tp_scores, fp_scores

    def reset(self):
        self.metric.reset()
        self.metric_35.reset()
        self._all_preds.clear()
        self._all_targets.clear()
