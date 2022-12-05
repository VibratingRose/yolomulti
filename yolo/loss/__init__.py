from .detloss import ComputeLoss
from .segloss import SegLoss, ComposeLoss, DiceLoss, BCEloss, FocalLoss

__all__ = ['ComputeLoss', 'SegLoss', 'DiceLoss', 'BCEloss', 'FocalLoss']
