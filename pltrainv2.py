import argparse
import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from yolo import tasks
from yolo.utils.general import check_dataset, check_file, check_yaml, yaml_load
from yolo.utils.register import Tasks

warnings.filterwarnings('ignore')

ROOT = Path("./")


def get_args(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', type=str, default='YoloSeg', help="choose the task")
    parser.add_argument(
        '--cfg', type=str, default="configs/seg.yaml", help='model.yaml path')
    parser.add_argument(
        '--data', type=str, default='data/electronNetProject.yaml', help='dataset.yaml path')
    parser.add_argument(
        '--hyp', type=str, default='configs/hyp/seg.hyp.yaml', help='hyperparameters path')
    parser.add_argument(
        '--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument(
        '--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
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
        '--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    # parser.add_argument(
    #     '--project', default=ROOT / 'runs/train', help='save to project/name')
    # parser.add_argument('--name', default='exp', help='save to project/name')
    # parser.add_argument(
    #     '--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument(
        '--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument(
        '--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')

    parser.add_argument(
        '--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def check_args():
    opt = get_args(True)
    opt.data = check_file(opt.data)
    opt.cfg = check_yaml(opt.cfg)
    opt.hyp = check_yaml(opt.hyp)

    hyp = yaml_load(opt.hyp)
    for k,v in hyp.items():
        setattr(opt, k, v)

    opt.data_dict = check_dataset(opt.data)  # check if None
    opt.nc = int(opt.data_dict['nc'])
    if opt.accumulate_grad_batches == -1:
        opt.accumulate_grad_batches = int(64 / opt.batch_size)
    opt.num_devices = len(opt.device.split(","))
    opt.hyp = hyp.copy()
    return opt


if __name__ == "__main__":
    opt = check_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
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
