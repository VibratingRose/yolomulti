import os
from functools import reduce

import numpy as np
import torch

from yolo.loss import ComputeLoss
from yolo.models.Builder import Model
from yolo.tools.va_batch import val_batch
from yolo.utils.dataloaders import create_dataloader
from yolo.utils.general import check_dataset, check_img_size, colorstr
from yolo.utils.metrics import ap_per_class
from yolo.utils.register import Tasks

from .base import Base


@Tasks.register
class YoloLightning(Base):
    def __init__(self, cfg, ch, nc, *args, **kwargs):
        super().__init__(cfg, ch, nc, *args, **kwargs)
        self.model = Model(cfg, ch=3, nc=nc)
        if "opt" in kwargs:
            self.opt = kwargs["opt"]
            self.config()

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        imgs, targets, _, _ = batch
        imgs = imgs / 255.0
        # inference
        preds = self(imgs)[:3]
        # caculate loss
        self.loss.set_device(self.device)
        loss, loss_items = self.loss(preds, targets)
        # log
        log_dict = {"box_loss": loss_items[0],
                    "obj_loss": loss_items[1],
                    "cls_loss": loss_items[2]}

        self.log_dict(log_dict, sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        # 这里相当于一个
        return val_batch(batch, self.model, self.loss, device=self.device)

    def validation_epoch_end(self, outputs):
        # outputs: [[(),()...], [(),()...],...]
        # each () is output coming from one image
        # each [(),()...] is output coming from one batch
        stats = reduce(lambda x,y: x+y, outputs)
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        map50 = 0.0
        map95 = 0.0
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir='.', names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map95 = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=self.nc)  # number of targets per class
        self.log('map50', map50, sync_dist=True, prog_bar=False, add_dataloader_idx=False)
        self.log('map95', map95, sync_dist=True, prog_bar=False, add_dataloader_idx=False)

    def train_dataloader(self):
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

    def val_dataloader(self):
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

    def save_opts(self):
        pass