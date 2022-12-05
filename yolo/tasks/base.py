from pathlib import Path
from typing import overload

import pytorch_lightning as pl
import torch
import yaml
from torch.optim import lr_scheduler

from yolo.models.Builder import Model
from yolo.utils.general import intersect_dicts
from yolo.utils.torch_utils import smart_optimizer


class Base(pl.LightningModule):
    def __init__(self, cfg, ch, nc, *args, **kwargs):
        super().__init__()
        self.model = Model(cfg, ch=3, nc=nc)
        if "opt" in kwargs:
            self.opt = kwargs["opt"]
    
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

    def training_epoch_end(self, outputs):
        if self.trainer.local_rank in [0, -1]:
            cfg_files = Path(self.trainer.log_dir)/"opt.yaml"
            if cfg_files.exists():
                pass
            else:
                with open(str(cfg_files), 'w', encoding='utf-8') as f:
                    yaml.dump(self.opt, f)


    def configure_optimizers(self):
        opt = self.opt
        hyp = self.hyp
        optimizer = smart_optimizer(self.model, opt.optimizer,
                                    opt.lr, opt.momentum, opt.weight_decay)

        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.01 * opt.lr)
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
        like model.{idx}.{some other info}, where idx means the main layer index
        in model.config.yaml like "yolo/configs/model/yolov5/yolov5s.yaml"
        """
        if len(freeze_list) == 1:
            freeze_list = list(range(freeze_list[0]))

        for k, v in self.model.named_parameters():
            if int(k.split(".")[1]) in freeze_list:
                v.requires_grad = False
        total = [v.requires_grad for k, v in self.model.named_parameters()]
        print(f"{sum(total)} of {len(total)} weights freezed.")

    def mannul_load_ckpt(self, ckpt_path, exclude=()):
        ckpt_path = Path(ckpt_path)
        if not (ckpt_path.exists() and ckpt_path.suffix in ['.pt', '.ckpt']):
            print("no pretrain weights loaded!")
            return
        weight = torch.load(str(ckpt_path), map_location='cpu')
        if 'ema' in weight or 'model' in weight:
            weight = (weight.get('ema') or weight.get('model')).state_dict()
        elif 'state_dict' in weight:
            weight = weight['state_dict']
        
        csd = intersect_dicts(weight, self.model.state_dict(), exclude=exclude)  # intersect
        print(f"{len(list(csd.keys()))} obj params loaded from pretrained models")
        self.model.load_state_dict(csd, strict=False)  # load