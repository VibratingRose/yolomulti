from pathlib import Path
from typing import overload

import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler

from yolo.models.Builder import Model
from yolo.utils.general import check_dataset, check_img_size, intersect_dicts
from yolo.utils.torch_utils import smart_optimizer


class Base(pl.LightningModule):
    def __init__(self, cfg, ch, nc, *args, **kwargs):
        super().__init__()
        self.model = Model(cfg, ch=3, nc=nc)
        if "opt" in kwargs:
            self.opt = kwargs["opt"]
            self.config()
    
    def config(self):
        opt = self.opt
        self.hyp = hyp = opt.hyp
        self.data_dict = check_dataset(opt.data)  # check if None
        names = self.data_dict['names']
        nc = int(self.data_dict['nc'])
        self.gs = max(int(self.model.stride.max()), 32)
        self.imgsz = check_img_size(opt.imgsz, self.gs, floor=self.gs * 2)

        # number of detection layers (to scale hyps)
        nl = self.model.model[-1].nl
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (self.imgsz / 640) ** 2 * 3 / nl
        # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing
        self.model.nc = nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.names = names

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    @overload
    def training_step(self, batch, batch_idx):
        pass

    @overload
    def validation_step(self, batch, batch_idx):
        pass

    @overload
    def train_dataloader(self):
        pass

    @overload
    def val_dataloader(self):
        pass

    def training_step_end(self, step_output):
        self.log('lr', self.lr_schedulers().get_last_lr()[0], sync_dist=True)

    def configure_optimizers(self):
        opt = self.opt
        hyp = self.hyp
        optimizer = smart_optimizer(self.model, opt.optimizer,
                                    hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.01 * hyp['lr0'])
        return [optimizer], [scheduler]

    def on_train_end(self):
        # 为了方便直接调用yolov5的部分函数
        last_name = f"{Path(self.trainer.log_dir)}/last.pt"
        torch.save({"model": self.model, 'opt': self.opt}, last_name)

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       optimizer_closure,
                       on_tpu=False,
                       using_native_amp=False,
                       using_lbfgs=False,
                       ):
        # 前300个step采用直线的warmp
        warmup_steps = self.opt.warmup_steps
        if warmup_steps > 0 and self.trainer.global_step <= warmup_steps:
            for pg in optimizer.param_groups:
                delta = (pg['initial_lr'] - pg['initial_lr'] / 100) / warmup_steps
                pg["lr"] = pg['initial_lr'] / 100 + delta * self.trainer.global_step
                if 'delta' not in pg.keys():
                    pg['delta'] = delta

        optimizer.step(closure=optimizer_closure)
        

    def mannul_freeze(self, freeze_list=[0]):
        """
        for model parsed by yolov5, eg. self.model, the model has the key
        like model.{idx}.{some other info}, where idx means the man layer index
        in model.config.yaml like "yolo/configs/model/yolov5/yolov5s.yaml"
        """
        if len(freeze_list) == 1:
            freeze_list = list(range(freeze_list[0]))

        for k, v in self.model.named_parameters():
            if int(k.split(".")[1]) in freeze_list:
                v.requires_grad = False
        total = [v.requires_grad for k, v in self.model.named_parameters()]
        print(f"{sum(total)} of {len(total)} weights freezed.")

    def mannul_load_ckpt(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not (ckpt_path.exists() and ckpt_path.suffix in ['.pt', '.ckpt']):
            return
        weight = torch.load(str(ckpt_path), map_location='cpu')
        if 'ema' in weight or 'model' in weight:
            weight = (weight.get('ema') or weight.get('model')).state_dict()
        elif 'state_dict' in weight:
            weight = weight['state_dict']
        exclude = ()
        csd = intersect_dicts(weight, self.model.state_dict(), exclude=exclude)  # intersect
        self.model.load_state_dict(csd, strict=False)  # load
