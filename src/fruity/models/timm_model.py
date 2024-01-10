from typing import Any, Tuple, Mapping

import timm
import torch

# from fairscale.nn import auto_wrap, checkpoint_wrapper, wrap
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchvision import transforms as T


def create_model(model: str, input_ch: int = 3, num_cls: int = 10) -> timm.models:
    """Load a model from timm.

    Args:
    ----
        model (str):    Name of model to load.
        input_ch (int): Number of input channels.
        num_cls (int):  Number of classes.
    """
    timm_model = timm.models.create_model(model, pretrained=True, in_chans=input_ch, num_classes=num_cls)
    return timm_model


class TIMMModule(LightningModule):
    """LightningModule for TIMM classification."""

    def __init__(
        self,
        net: timm.models,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """LightningModule for TIMM classification.

        Args:
        ----
            net (timm.models):      Model to train.
            optimizer (Optimizer):  Optimizer to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=net.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=net.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=net.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.predict_transform = torch.nn.Sequential(
            T.Resize([100, 100]),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
        ----
            x (torch.Tensor): Input tensor.

        Returns:
        -------
            torch.Tensor: Output tensor
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Reset metrics at beginning of training."""
        self.val_acc_best.reset()

    def step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network and calculate loss.

        Args:
        ----
            batch (Any): Input batch.

        Returns:
        -------
            Tuple(torch.Tensor, torch.Tensor, torch.Tensor): Tuple of loss, predictions and targets.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, torch.Tensor]:
        """Train on batch.

        Args:
        ----
            batch (Any): Input batch.
            batch_idx (int): Index of batch.

        Returns:
        -------
            Mapping[str, torch.Tensor]: Dict with loss, predictions and targets.
        """
        loss, preds, targets = self.step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, torch.Tensor]:
        """Validate on batch.

        Args:
        ----
            batch (Any): Input batch.
            batch_idx (int): Index of batch.

        Returns:
        -------
            Mapping[str, torch.Tensor]: Dict with loss, predictions and targets.
        """
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("hp_metric", self.val_loss, sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self) -> None:
        """Log best so far validation accuracy."""
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, torch.Tensor]:
        """Test on batch.

        Args:
        ----
            batch (Any): Input batch.
            batch_idx (int): Index of batch.

        Returns:
        -------
            Mapping[str, torch.Tensor]: Dict with loss, predictions and targets.
        """
        loss, preds, targets = self.step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self) -> Mapping[str, torch.optim.Optimizer]:
        """Return optimizer."""
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }
