from typing import Any, Dict, Optional

import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchgeo import datamodules
from torchgeo.datasets import RESISC45, EuroSAT


class CutMix(nn.Module):
    """Wrapper around kornia.augmentation.RandomCutMix to postprocess labels to onehot"""

    def __init__(self, num_classes, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        kwargs["num_mix"] = 1  # only supports mixups=1 for now
        self.transform = K.RandomCutMix(*args, **kwargs)

    @staticmethod
    def to_onehot(yp, y, num_classes):
        targets = F.one_hot(y, num_classes=num_classes)
        targets_perm = F.one_hot(yp[0, :, 1].to(torch.long), num_classes=num_classes)
        lam = yp[0, :, 2]
        lam = lam[:, None]
        labels = targets * lam + targets_perm * (1 - lam)
        labels = torch.clip(labels, min=0.0, max=1.0)
        return labels

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xp, yp = self.transform(x, y)
        # random cutmix didn't happen so still need to onehot the labels
        if yp.ndim != 3:
            yp = F.one_hot(y, self.num_classes).to(torch.float)
        # random cutmix happened
        else:
            yp = self.to_onehot(yp, y, self.num_classes)
        return xp, yp


class RESISC45DataModule(datamodules.RESISC45DataModule):

    resize_transform = T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentations = K.AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.RandomErasing(p=0.1),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=["input"],
        )
        if "cutmix" in kwargs:
            print("Using CutMix Augmentation")
            self.cutmix = CutMix(num_classes=45, height=224, width=224, p=0.5)
        else:
            self.cutmix = None

    def resize(self, sample):
        sample["image"] = self.resize_transform(sample["image"])
        return sample

    def on_after_batch_transfer(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        """Apply batch augmentations after batch is transferred to the device.
        Args:
            batch: mini-batch of data
            batch_idx: batch index
        Returns:
            augmented mini-batch
        """
        if (
            hasattr(self, "trainer")
            and self.trainer is not None
            and hasattr(self.trainer, "training")
            and self.trainer.training
        ):
            x, y = batch["image"], batch["label"]
            x = self.augmentations(x)

            if self.cutmix is not None:
                x, y = self.cutmix(x, y)

            batch["image"], batch["label"] = x, y
        else:
            if self.cutmix is not None:
                batch["label"] = F.one_hot(batch["label"], num_classes=45)
        return batch

    def setup(self, stage: Optional[str] = None) -> None:
        transforms = T.Compose([self.preprocess, self.resize])
        self.train_dataset = RESISC45(self.root_dir, "train", transforms=transforms)
        self.val_dataset = RESISC45(self.root_dir, "val", transforms=transforms)
        self.test_dataset = RESISC45(self.root_dir, "test", transforms=transforms)

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        return plt.figure()


class EuroSATDataModule(datamodules.EuroSATDataModule):

    # resize_transform = T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentations = K.AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.RandomErasing(p=0.1),
            # K.ColorJitter(
            #    p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            # ),
            data_keys=["input"],
        )
        if "cutmix" in kwargs:
            print("Using CutMix Augmentation")
            self.cutmix = CutMix(num_classes=10, height=64, width=64, p=0.5)
        else:
            self.cutmix = None

    def resize(self, sample):
        sample["image"] = sample["image"].float()
        sample["image"] = self.norm(sample["image"])
        # sample["image"] = self.resize_transform(sample["image"])
        return sample

    def on_after_batch_transfer(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        """Apply batch augmentations after batch is transferred to the device.
        Args:
            batch: mini-batch of data
            batch_idx: batch index
        Returns:
            augmented mini-batch
        """
        if (
            hasattr(self, "trainer")
            and self.trainer is not None
            and hasattr(self.trainer, "training")
            and self.trainer.training
        ):
            x, y = batch["image"], batch["label"]
            x = self.augmentations(x)

            if self.cutmix is not None:
                x, y = self.cutmix(x, y)

            batch["image"], batch["label"] = x, y
        else:
            if self.cutmix is not None:
                batch["label"] = F.one_hot(batch["label"], num_classes=10)
        return batch

    def setup(self, stage: Optional[str] = None) -> None:
        transforms = T.Compose([self.preprocess, self.resize])
        self.train_dataset = EuroSAT(self.root_dir, "train", transforms=transforms)
        self.val_dataset = EuroSAT(self.root_dir, "val", transforms=transforms)
        self.test_dataset = EuroSAT(self.root_dir, "test", transforms=transforms)

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        return plt.figure()
