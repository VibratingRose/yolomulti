from yolo.datasets.segdatasets import get_dataloader as segloader
from yolo.loss import SegLoss
from yolo.metrics.seg_metrics import iou_fg
from yolo.utils.general import check_dataset, check_img_size

from .base import Base
from yolo.utils.register import Tasks


@Tasks.register
class YoloSeg(Base):
    def __init__(self, cfg, ch, nc, *args, **kwargs):
        super().__init__(cfg, ch, nc, *args, **kwargs)
        if hasattr(self, 'opt'):
            self.opt = kwargs['opt']
            self.config()
            bce_weight = self.opt.bce_weight if hasattr(self.opt, 'bce_weight') else 0.7
            self.seg_loss = SegLoss(bce_weight=bce_weight)
            self.mannul_load_ckpt(self.opt.weights)
            self.mannul_freeze(self.opt.freeze)

    def config(self):
        opt = self.opt
        self.hyp = hyp = opt.hyp
        self.data_dict = check_dataset(opt.data)  # check if None
        names = self.data_dict['names']
        nc = int(self.data_dict['nc'])
        # self.gs = max(int(self.model.stride.max()), 32)
        # self.imgsz = check_img_size(opt.imgsz, self.gs, floor=self.gs * 2)

        # number of detection layers (to scale hyps)
        # nl = self.model.model[-1].nl
        # hyp['box'] *= 3 / nl  # scale to layers
        # hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        # hyp['obj'] *= (self.imgsz / 640) ** 2 * 3 / nl
        # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing
        self.model.nc = nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.names = names

    def training_step(self, batch, batch_idx):
        assert 0 < self.opt.seg_weight and self.opt.seg_weight <= 1, "seg_weight error"
        seg_loss = self.seg_forward_step(batch, batch_idx)['loss']
        return {'loss': seg_loss}

    def train_dataloader(self):
        opt = self.opt
        seg_train = self.data_dict['seg_train']
        seg_dataloader = segloader(
            seg_train, opt.batch_size, opt.workers, "train")
        return seg_dataloader

    def validation_step(self, batch, batch_idx):
        self.seg_forward_step(batch, batch_idx, acc=True)

    def val_dataloader(self):
        opt = self.opt
        seg_val_loader = segloader(
            self.data_dict['seg_val'], opt.batch_size, opt.workers, "val")
        return seg_val_loader

    def seg_forward_step(self, batch, batch_idx, acc=False):
        imgs, labels = batch
        imgs = imgs / 255.0
        preds = self(imgs)
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        loss = self.seg_loss(preds, labels)
        self.log("seg_loss", loss, prog_bar=True, sync_dist=True)

        if acc:
            acc = iou_fg(preds, labels)
            self.log("seg_acc", acc, prog_bar=True, sync_dist=True)
        return {'loss': loss}
