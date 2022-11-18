import torch

from yolo.datasets.segdatasets import get_dataloader as segloader
from yolo.loss import ComputeLoss, SegLoss
from yolo.metrics.seg_metrics import iou_fg
from yolo.utils.general import check_dataset, check_img_size
from yolo.utils.register import Tasks

from .yolo_lightning import YoloLightning


@Tasks.register
class MultiTaskv2(YoloLightning):
    def __init__(self, cfg, ch, nc, *args, **kwargs):
        super().__init__(cfg, ch, nc, *args, **kwargs)
        if hasattr(self, 'opt'):
            self.opt = kwargs['opt']
            self.config()
            bce_weight = self.opt.bce_weight if hasattr(self.opt, 'bce_weight') else 0.7
            self.seg_loss = SegLoss(bce_weight)
            self.mannul_load_ckpt(self.opt.weights)
            self.mannul_freeze(self.opt.freeze)
        # self.loss_scale = nn.Parameter(torch.tensor([-0.5] * 2, device=self.device))

    def training_step(self, batch, batch_idx):
        det_batch, seg_batch = batch
        assert 0<self.opt.seg_weight and self.opt.seg_weight <=1, "seg_weight error"

        det_loss = super().training_step(det_batch, batch_idx)['loss']
        seg_loss = self.seg_training_step(seg_batch, batch_idx)['loss']

        losses = (1 - self.opt.seg_weight) * det_loss + self.opt.seg_weight * seg_loss
        # losses = torch.tensor([det_loss, seg_loss], requires_grad=True)
        # # losses = (losses/(2*self.loss_scale.exp())+self.loss_scale/2).sum()
        # weight = 1 / losses.detach().softmax(-1)
        # losses = (weight * losses).sum() / weight.sum()
        return {'loss': losses}
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            output = super().validation_step(batch, batch_idx)
            return output
        elif dataloader_idx == 1:
            self.seg_training_step(batch, batch_idx, acc=True)

    def seg_training_step(self, batch, batch_idx, acc=False):
        imgs, labels = batch
        imgs = imgs / 255.0
        preds = self(imgs)[-1]
        loss = self.seg_loss(preds, labels)
        self.log("seg_loss", loss, prog_bar=True, sync_dist=True, add_dataloader_idx=False)

        if acc:
            acc = iou_fg(preds, labels)
            self.log("seg_acc", acc, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        super().validation_epoch_end(outputs[0])

    def train_dataloader(self):
        # det_dataloader = self.det_train_dataloader()
        det_dataloader = super().train_dataloader()
        opt = self.opt
        seg_train = self.data_dict['seg_train']
        seg_dataloader = segloader(
            seg_train, opt.batch_size, opt.workers, "train")
        return det_dataloader, seg_dataloader

    def val_dataloader(self):
        # det_val_loader = self.det_val_dataloader()
        det_val_loader = super().val_dataloader()
        seg_val_loader = self.seg_val_dataloader()
        return det_val_loader, seg_val_loader
    
    def seg_val_dataloader(self):
        opt = self.opt
        seg_val_loader = segloader(
            self.data_dict['seg_val'], opt.batch_size, opt.workers, "val")
        return seg_val_loader

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