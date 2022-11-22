import argparse
import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from yolo import tasks
from yolo.utils.general import (check_dataset, check_file, check_yaml,
                                increment_path, yaml_load)
from yolo.utils.register import Tasks

warnings.filterwarnings('ignore')

ROOT = Path("./")


def get_args(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', type=str, default='YoloSeg', help="choose the task")
    parser.add_argument(
        '--weights', type=str, default='weights/yolov5_state_dict.ckpt', help='initial weights path')
    parser.add_argument(
        '--cfg', type=str, default="configs/detseg.yolov1.yaml", help='model.yaml path')
    parser.add_argument(
        '--data', type=str, default='data/electronNetProject.yaml', help='dataset.yaml path')
    parser.add_argument(
        '--hyp', type=str, default='yolo/configs/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument(
        '--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument(
        '--batch-size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument(
        '--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument(
        '--rect', action='store_true', help='rectangular training')
    parser.add_argument(
        '--resume', nargs='?', const=True, default=False, help='resume most recent training')

    parser.add_argument('--warmup_steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument(
        '--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument(
        '--noplots', action='store_true', help='save no plot files')

    parser.add_argument(
        '--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument(
        '--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument(
        '--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        '--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument(
        '--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument(
        '--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument(
        '--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument(
        '--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument(
        '--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument(
        '--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument(
        '--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument(
        '--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument(
        '--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True,
                        default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1,
                        help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str,
                        default='latest', help='Version of dataset artifact to use')
    parser.add_argument('--accumulate_grad_batches', type=int,
                        default=-1, help='accumulate_grad_batches')

    # for multitasks
    parser.add_argument('--seg_weight', type=float, default=1.0,
                        help='the weight for segmentation loss')
    parser.add_argument('--det_weight', type=float, default=1.0,
                        help='the weight for segmentation loss')
    parser.add_argument('--bce_weight', type=float,
                        default=0.7, help="the bce weight for seg_loss")
    parser.add_argument('--split_train', action='store_true',
                        help='to train det and seg in different batch')

    parser.add_argument('--backbone_lr_ratio', type=float,
                        default=0, help='to train det and seg in different batch')
    parser.add_argument('--backbone_index', type=int, default=0,
                        help='to train det and seg in different batch')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def check_args():
    opt = get_args(True)
    opt.data = check_file(opt.data)
    opt.cfg = check_yaml(opt.cfg)
    opt.hyp = check_yaml(opt.hyp)
    opt.weights = str(opt.weights)  # opt.weights 原本是 pathlib.PosixPath
    opt.project = str(opt.project)
    assert len(opt.cfg) or len(
        opt.weights), 'either --cfg or --weights must be specified'
    if opt.name == 'cfg':
        opt.name = Path(opt.cfg).stem  # use model.yaml as name
    opt.save_dir = str(increment_path(
        Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    meta = {
        'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
        'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
        'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
        'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
        'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
        'box': (1, 0.02, 0.2),  # box loss gain
        'cls': (1, 0.2, 4.0),  # cls loss gain
        'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
        'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
        'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
        'iou_t': (0, 0.1, 0.7),  # IoU training threshold
        'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
        'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
        # focal loss gamma (efficientDet default gamma=1.5)
        'fl_gamma': (0, 0.0, 2.0),
        'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
        'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
        'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
        'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
        'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
        # image perspective (+/- fraction), range 0-0.001
        'perspective': (0, 0.0, 0.001),
        'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
        'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
        'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
        'mixup': (1, 0.0, 1.0),  # image mixup (probability)
        'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

    hyp = yaml_load(opt.hyp)
    hyp["anchors"] = hyp.get("anchors", 3)
    opt.hyp = hyp.copy()
    opt.hyp["lr0"] = opt.lr

    if opt.noautoanchor:
        del hyp['anchors'], meta['anchors']

    opt.noval = True
    opt.nosave = True

    opt.data_dict = check_dataset(opt.data)  # check if None
    opt.nc = int(opt.data_dict['nc'])
    if opt.accumulate_grad_batches == -1:
        opt.accumulate_grad_batches = int(64 / opt.batch_size)
    opt.num_devices = len(opt.device.split(","))
    return opt


if __name__ == "__main__":
    opt = check_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    opt.num_devices = len(opt.device.split(','))
    print(opt.task)
    model = Tasks[opt.task](opt.cfg, 3, 3, opt=opt)
    seed_everything(3407)
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator='gpu',
        devices=opt.num_devices,
        accumulate_grad_batches=opt.accumulate_grad_batches,
        sync_batchnorm=opt.num_devices > 1,
        strategy="ddp" if opt.num_devices > 1 else None
    )
    trainer.fit(model)
