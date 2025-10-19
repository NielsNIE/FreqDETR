import torch

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5*w), (cy - 0.5*h), (cx + 0.5*w), (cy + 0.5*h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1)/2, (y0 + y1)/2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou_xyxy(boxes1, boxes2):
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(0) * (boxes1[:,3]-boxes1[:,1]).clamp(0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(0) * (boxes2[:,3]-boxes2[:,1]).clamp(0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[...,0]*wh[...,1]
    union = area1[:,None] + area2 - inter + 1e-6
    return inter/union

def generalized_box_iou(boxes1, boxes2):
    iou = box_iou_xyxy(boxes1, boxes2)
    lt_c = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    c_area = wh_c[...,0]*wh_c[...,1] + 1e-6

    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(0) * (boxes1[:,3]-boxes1[:,1]).clamp(0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(0) * (boxes2[:,3]-boxes2[:,1]).clamp(0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[...,0]*wh[...,1]
    union = area1[:,None] + area2 - inter + 1e-6
    giou = iou - (c_area - union)/c_area
    return giou