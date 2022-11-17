import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import tqdm

from yolo.models.Builder import Model
from yolo.datasets.segdatasets import get_dataloader
from yolo.metrics import iou_fg


class Validation:
    def __init__(self, model_path=None, cfg=None, save_pt=False):
        weights, ckpt = self.load_weight(model_path)
        if save_pt:
            save_path = Path(ckpt).parent / f"{Path(ckpt).stem}.pt"
        else:
            save_path = None
        self.model = self.load_model(cfg, weights, save_path)

    def load_weight(self, path):
        if Path(path).suffix != ".ckpt":
            ckpt = list(Path(path).rglob("*.ckpt"))[0]
        else:
            ckpt = path
        ckpt = str(ckpt)
        weights = torch.load(ckpt, map_location='cpu')['state_dict']
        new_weights = OrderedDict()
        for k, v in weights.items():
            new_weights[k.split('.', 1)[1]] = v
        return new_weights, ckpt

    def load_model(self, cfg, weights, path=None, device='cpu'):
        model = Model(cfg, ch=3, nc=3)
        model.load_state_dict(weights)
        model.eval()
        model.to(device)
        if path is not None:
            path = str(path)
            torch.save({'model':model}, path)
        return model


    def val_seg(self, data_path, reduction="mean"):
        data = get_dataloader(data_path, batch_size=1, num_workers=1, mode="val")
        ious = []
        self.model.cuda()
        print(f"start to validate the segmentations...")
        pbar = tqdm.tqdm(data)
        for i,(img, labels) in enumerate(pbar):
            img = img.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                pred = self.model(img)[-1]
                iou = iou_fg(pred, labels)
                ious.append(iou.detach().cpu().item())
        ious = np.array(ious)
        if reduction=="mean":
            ious = ious.mean()

        return ious

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--version', type=str, default='1')
    args.add_argument('--save_pt', action='store_true')
    opt = args.parse_args()
    data_path = "/work/datasets/segmentation/val_1105.txt"
    cfg_path = "configs/detseg.yolo.v1.yaml"
    # cfg_path = "configs/detseg.repyolo.v1.yaml"
    ckpt_path = f"lightning_logs/version_{opt.version}"
    val = Validation(ckpt_path, cfg_path, opt.save_pt)
    ious = val.val_seg(data_path)
    print(ious)
