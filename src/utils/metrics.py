import torch
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

@torch.no_grad()
def ap50_greedy(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    # pred_boxes: [P,4] (cxcywh, 0..1), pred_scores: [P]
    # gt_boxes: [G,4] (cxcywh, 0..1)
    device = pred_boxes.device
    if gt_boxes.numel()==0 and pred_boxes.numel()==0:
        return 1.0, 1.0, 1.0
    if gt_boxes.numel()==0:
        return 0.0, 0.0, 0.0
    if pred_boxes.numel()==0:
        return 0.0, 0.0, 0.0

    pb = box_cxcywh_to_xyxy(pred_boxes)
    gb = box_cxcywh_to_xyxy(gt_boxes)

    used = torch.zeros(len(gb), dtype=torch.bool, device=device)
    order = torch.argsort(pred_scores, descending=True)
    tp, fp = 0, 0
    for i in order:
        box = pb[i:i+1]
        ious = generalized_box_iou(box, gb)[0]
        j = torch.argmax(ious)
        if ious[j] >= iou_thresh and not used[j]:
            tp += 1; used[j] = True
        else:
            fp += 1
    fn = (~used).sum().item()
    prec = tp / max(tp+fp, 1)
    rec  = tp / max(tp+fn, 1)
    f1 = (2*prec*rec)/(prec+rec+1e-6)
    return f1, prec, rec