from typing import Optional

import torch.nn as nn
import torchgeo.datamodules
import torchgeo.trainers
import torchvision
import torchvision.transforms as T
from torchgeo.datasets import RESISC45


class RESISC45DataModule(torchgeo.datamodules.RESISC45DataModule):

    resize_transform = T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def resize(self, sample):
        sample["image"] = self.resize_transform(sample["image"])
        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        transforms = T.Compose([self.preprocess, self.resize])
        self.train_dataset = RESISC45(self.root_dir, "train", transforms=transforms)
        self.val_dataset = RESISC45(self.root_dir, "val", transforms=transforms)
        self.test_dataset = RESISC45(self.root_dir, "test", transforms=transforms)


class RESISC45ClassificationTask(torchgeo.trainers.ClassificationTask):
    def config_model(self) -> None:
        """Configures the model based on kwargs parameters passed to the constructor."""
        classification_model = self.hparams["classification_model"]
        num_classes = self.hparams["num_classes"]

        if "resnet" in classification_model:
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif "vit" in classification_model:
            self.model = torchvision.models.vit_b_16(pretrained=True)
            self.model.heads.head = nn.Linear(
                self.model.heads.head.in_features, num_classes
            )
        elif "convnext" in classification_model:
            self.model = torchvision.models.convnext_base(pretrained=True)
            self.model.classifier[2] = nn.Linear(
                self.model.classifier[2].in_features, num_classes
            )
        else:
            raise ValueError(
                f"Model type '{classification_model}' is not a valid model."
            )
