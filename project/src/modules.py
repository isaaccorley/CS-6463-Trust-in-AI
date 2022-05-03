from typing import Any, Dict, cast

import torch
from torchmetrics import Accuracy, FBetaScore, MetricCollection
from torchgeo import trainers


class CustomMultiLabelClassificationTask(trainers.MultiLabelClassificationTask):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.hyperparams = cast(Dict[str, Any], self.hparams)
        self.config_task()
        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    num_classes=self.hyperparams["num_classes"],
                    average="micro",
                    multiclass=False,
                ),
                "AverageAccuracy": Accuracy(
                    num_classes=self.hyperparams["num_classes"],
                    average="macro",
                    multiclass=False,
                ),
                "F1Score": FBetaScore(
                    num_classes=self.hyperparams["num_classes"],
                    beta=1.0,
                    average="micro",
                    multiclass=False,
                ),
            },
            prefix="val_",
        )
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.train_metrics = None

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Compute and return the training loss.
        Args:
            batch: the output of your DataLoader
        Returns:
            training loss
        """
        batch = args[0]
        x = batch["image"]
        y = batch["label"]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return cast(torch.Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        return None
