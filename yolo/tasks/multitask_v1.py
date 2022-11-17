import os
from functools import reduce

import numpy as np
import torch

from yolo.datasets.segdatasets import get_dataloader as segloader
from yolo.loss import ComputeLoss, SegLoss
from yolo.metrics.seg_metrics import iou_fg
from yolo.tools.va_batch import val_batch
from yolo.utils.dataloaders import create_dataloader
from yolo.utils.general import check_dataset, check_img_size, colorstr
from yolo.utils.metrics import ap_per_class
from yolo.utils.register import Tasks

from .base import Base


@Tasks.register
class MultiTask(Base):
    def __init__(self, cfg, ch, nc, *args, **kwargs):
        super().__init__(cfg, ch, nc, *args, **kwargs)
        if hasattr(self, 'opt'):
            self.opt = kwargs['opt']
            self.config()
            bce_weight = self.opt.bce_weight if hasattr(self.opt, 'bce_weight') else 0.7
            self.seg_loss = SegLoss(bce_weight)
            self.mannul_load_ckpt(self.opt.weights)
            self.mannul_freeze(self.opt.freeze)

    def training_step(self, batch, batch_idx):
        det_batch, seg_batch = batch
        assert 0<self.opt.seg_weight and self.opt.seg_weight <=1, "seg_weight error"

        det_loss = self.det_training_step(det_batch, batch_idx)['loss']
        seg_loss = self.seg_training_step(seg_batch, batch_idx)['loss']
        # total_loss = (1 - self.opt.seg_weight) * det_loss + self.opt.seg_weight * seg_loss
        # auto 
        weight = torch.tensor([seg_loss, det_loss]).detach().softmax(-1)
        total_loss = weight[0] * det_loss + weight[1] * seg_loss
        return {'loss': total_loss}
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        # detection validation, 
        if dataloader_idx == 0:
            output = val_batch(batch, self.model, self.loss, device=self.device)
            return output
        elif dataloader_idx == 1:
        # segmentation validation, which does not need to return 
            self.seg_training_step(batch, batch_idx, acc=True)

    def det_training_step(self, batch, batch_idx):
        imgs, targets, _, _ = batch
        imgs = imgs / 255.0

        preds = self(imgs)[:3]
        self.loss.set_device(self.device)
        loss, loss_items = self.loss(preds, targets)

        log_dict = {"box_loss": loss_items[0],
                    "obj_loss": loss_items[1],
                    "cls_loss": loss_items[2]}

        self.log_dict(log_dict, sync_dist=True, prog_bar=True)
        return {"loss": loss}

    def seg_training_step(self, batch, batch_idx, acc=False):
        imgs, labels = batch
        imgs = imgs / 255.0
        preds = self(imgs)[-1]
        loss = self.seg_loss(preds, labels)
        self.log("seg_loss", loss, prog_bar=True, sync_dist=True)

        if acc:
            acc = iou_fg(preds, labels)
            self.log("seg_acc", acc, prog_bar=True, sync_dist=True)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        # outputs: [[[(),()...], [(),()...],...], []]
        # each list of the first layer is crossponding to the dataloader 
        # each () is output coming from one image
        # each [(),()...] is output coming from one batch
        stats = reduce(lambda x,y: x+y, outputs[0])
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        map50 = 0.0
        map95 = 0.0
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir='.', names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map95 = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=self.nc)  # number of targets per class
        self.log('map50', map50, sync_dist=True, prog_bar=True)
        self.log('map95', map95, sync_dist=True, prog_bar=False)

    def train_dataloader(self):
        det_dataloader = self.det_train_dataloader()
        opt = self.opt
        seg_train = self.data_dict['seg_train']
        seg_dataloader = segloader(
            seg_train, opt.batch_size, opt.workers, "train")
        return det_dataloader, seg_dataloader

    def val_dataloader(self):
        det_val_loader = self.det_val_dataloader()
        seg_val_loader = self.seg_val_dataloader()
        return det_val_loader, seg_val_loader
    
    def seg_val_dataloader(self):
        opt = self.opt
        seg_val_loader = segloader(
            self.data_dict['seg_val'], opt.batch_size, opt.workers, "val")
        return seg_val_loader

    def det_train_dataloader(self):
        opt = self.opt
        single_cls = False

        train_loader, _ = \
            create_dataloader(self.data_dict['train'],
                              self.imgsz,
                              opt.batch_size,
                              self.gs,
                              single_cls,
                              hyp=self.hyp,
                              augment=True,
                              cache=None if opt.cache == 'val' else opt.cache,
                              rect=opt.rect,
                              rank=int(os.getenv('LOCAL_RANK', -1)),
                              workers=opt.workers,
                              image_weights=opt.image_weights,
                              quad=opt.quad,
                              prefix=colorstr('train: '),
                              shuffle=True)
        return train_loader

    def det_val_dataloader(self):
        opt = self.opt
        single_cls = False
        noval = False
        val_loader = create_dataloader(self.data_dict['val'],
                                       self.imgsz,
                                       opt.batch_size,
                                       self.gs,
                                       single_cls,
                                       hyp=self.hyp,
                                       cache=None if noval else opt.cache,
                                       rect=False,
                                       rank=int(os.getenv('LOCAL_RANK', -1)),
                                       workers=opt.workers,
                                       pad=0,
                                       prefix=colorstr('val: '))[0]
        return val_loader

    def config(self):
        opt = self.opt
        self.hyp = hyp = opt.hyp
        self.data_dict = check_dataset(opt.data)  # check if None
        self.names = self.data_dict['names']
        self.nc = int(self.data_dict['nc'])
        self.gs = max(int(self.model.stride.max()), 32)
        self.imgsz = check_img_size(opt.imgsz, self.gs, floor=self.gs * 2)

        # number of detection layers (to scale hyps)
        nl = self.model.model[-1].nl
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= self.nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (self.imgsz / 640) ** 2 * 3 / nl
        # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.names = self.names
        self.loss = ComputeLoss(self.model)  # init loss class