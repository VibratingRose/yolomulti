import pickle
import random
from pathlib import Path

import albumentations as A
import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Dilation:
    def __init__(self, ks=4, iter=1, p=0.5):
        self.ks = ks
        self.iter = iter
        self.p = p

    def __call__(self, image=None, mask=None):
        assert mask is not None, "mask is none, please check"
        if random.random() < self.p:
            kernel = np.ones((self.ks, self.ks), np.uint8)
            mask_ = cv2.dilate(mask, kernel, iterations=1)
        return {"image": image, "mask": mask}

train_aug = A.Compose([
    # A.RandomResizedCrop(576, 576),
    A.RandomCrop(544, 544),
    A.Flip(),
    A.RandomRotate90(p=1.0),
    A.GaussNoise(p=0.2),
    A.ShiftScaleRotate(p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=.1),
        # A.RandomGridShuffle(grid=(2, 2))
    ], p=0.3),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.RandomBrightnessContrast(),
    ], p=0.2),
    A.HueSaturationValue(p=0.3),
    # A.ChannelShuffle(p=0.5),
    Dilation(p=1)
])

val_aug = A.Compose([
    A.RandomCrop(p=1, height=544, width=960),
    Dilation(p=1),
    A.Flip(),
])


class BaseDataset(Dataset):
    def __init__(self, annpath, imgsz=(640, 640), mode='train', aug=True):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.lb_ignore = -100
        aug = train_aug if mode == 'train' else val_aug
        self.aug = aug if aug else None
        self.pairs = self.get_images_path(annpath)
        self.imgsz = imgsz if isinstance(
            imgsz, (tuple, list)) else [imgsz, imgsz]

        # load images and labels from lmdb
        self.txn = None
        try:
            dbpath = Path(annpath) / "dbData"
            if dbpath.exists():
                self.db = lmdb.open(dbpath, subdir=True,
                                    map_size=1073741824 * 250, readonly=True,
                                    meminit=False, map_async=True)
                self.txn = self.begin()
        except:
            pass

    def __getitem__(self, idx):
        img, mask, _ = self.get_image(idx)
        if self.aug is not None:
            aug = self.aug(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
        inh, inw = self.imgsz
        if self.mode == 'train' and img.shape[0] != inh and img.shape[1] != inw:
            img = cv2.resize(
                img, (inw, inh), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(
                mask, (inw, inh), interpolation=cv2.INTER_AREA)
        img = img.transpose(2, 0, 1)
        mask = mask[None, ...]
        img = torch.tensor(img).float()
        mask = torch.tensor(mask).float()
        return img, mask

    def get_image(self, idx):
        impth, lbpth = self.pairs[idx]
        if self.txn is not None:
            stem = Path(impth).stem
            v = self.txn.get(stem.encode())
            img, mask, _ = pickle.loads(v)
        else:
            img = cv2.imread(impth)[:, :, ::-1].copy()
            mask = cv2.imread(lbpth, 0)
        return img, mask, impth

    def __len__(self):
        return len(self.pairs)

    def get_images_path(self, annpath):
        pairs = Path(annpath).read_text().splitlines()
        pairs = [l.split(",") for l in pairs]
        return pairs


def get_dataloader(file_path=None, batch_size=8, num_workers=8, mode="train"):
    assert mode in ('train', "val"), f"mode type: {mode} does not support!"
    file_path = Path(file_path).expanduser()
    data = BaseDataset(file_path, mode=mode)
    dl = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=mode == "train",
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dl
