from yolo import loss
from yolo.datasets.segdatasets import get_dataloader as segloader
from yolo.loss import SegLoss
from yolo.metrics.seg_metrics import iou_fg
from yolo.utils.general import check_dataset
from yolo.utils.register import Loss, Tasks

from .base import Base


@Tasks.register
class YoloSeg(Base):
    def __init__(self, cfg, ch, nc, *args, **kwargs):
        super().__init__(cfg, ch, nc, *args, **kwargs)
        if hasattr(self, 'opt'):
            self.opt = kwargs['opt']
            self.hyp = self.opt.hyp
            self.config()
            self.loss_configure()
            self.mannul_load_ckpt(self.opt.pretrain)
            self.mannul_freeze(self.opt.freeze)

    def loss_configure(self):
        loss = self.opt.loss
        type = loss.pop('type')
        self.compute_loss = Loss[type](**loss)

    def config(self):
        opt = self.opt
        self.data_dict = check_dataset(opt.data)  # check if None
        names = self.data_dict['names']
        nc = int(self.data_dict['nc'])
        self.model.nc = nc  # attach number of classes to model
        self.model.hyp = None  # attach hyperparameters to model
        self.model.names = names

    def training_step(self, batch, batch_idx):
        loss = self.seg_forward_step(batch, batch_idx)['loss']
        return {'loss': loss}

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
        loss = self.compute_loss(preds, labels)
        self.log("loss/seg_loss", loss, prog_bar=True, sync_dist=True)

        if acc:
            acc = iou_fg(preds, labels)
            self.log("metrics/seg_acc", acc, prog_bar=True, sync_dist=True)
        return {'loss': loss}
