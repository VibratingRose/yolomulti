from pathlib import Path

import torch

from yolo.tools.val import process_batch
from yolo.utils.general import non_max_suppression, scale_coords, xywh2xyxy


def val_batch(batch, model, lossfunc=None, device=None, conf_thres=0.001, iou_thres=0.6, **kwargs):
    imgs, targets, paths, shapes = batch
    nb, _, height, width = imgs.shape
    nc = model.nc

    # train
    training = lossfunc is not None
    augment = "augment" in kwargs and kwargs["augment"]

    imgs = imgs / 255.0
    out, train_out = model(imgs)[:2] if training else model(imgs, augment=augment, val=True)

    if lossfunc is not None:
        loss = torch.zeros(3, device=device)
        lossfunc.set_device(device)
        loss += lossfunc([x.float() for x in train_out], targets)[1]  # box, obj, cls

    # dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # jdict, , ap, ap_class = [], [], [], []
    stats = []

    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
    # some configs need to change.
    single_cls = False
    lb = []
    #
    out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
    for si, pred in enumerate(out):
        labels = targets[targets[:, 0] == si, 1:]
        nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
        path, shape = Path(paths[si]), shapes[si][0]
        correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init

        if npr == 0:
            if nl:
                stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
            continue

        predn = pred.clone()
        scale_coords(imgs[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(imgs[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)

        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
    return stats
