import random
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import _update_n
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

"""
progress bar with time
"""


class TimeFilterProgressBar(TQDMProgressBar):
    def __init__(
        self, keep_keywords: list = None, remove_keywords: list = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.keep_keywords = keep_keywords
        self.remove_keywords = remove_keywords

    def filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if self.keep_keywords:
            metrics = {
                k: v
                for k, v in metrics.items()
                if any(keyword in k for keyword in self.keep_keywords)
            }
        if self.remove_keywords:
            metrics = {
                k: v
                for k, v in metrics.items()
                if not any(keyword in k for keyword in self.remove_keywords)
            }
        return metrics

    def add_time(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["time"] = current_time
        return metrics

    def add_learning_rate(
        self, metrics: Dict[str, Any], trainer: "pl.Trainer"
    ) -> Dict[str, Any]:
        learning_rate = trainer.optimizers[0].param_groups[0]["lr"]
        metrics["lr"] = learning_rate
        return metrics

    def sort_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        # sort it in the order of train val time
        sorted_metrics = {}
        for keyword in ["v_num", "train", "val", "lr", "time"]:
            for k, v in metrics.items():
                if keyword in k:
                    sorted_metrics[k] = v
        return sorted_metrics

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            metrics = self.get_metrics(trainer, pl_module)
            metrics = self.filter_metrics(metrics)

            metrics = self.add_time(metrics)
            metrics = self.add_learning_rate(metrics, trainer)

            metrics = self.sort_metrics(metrics)

            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(metrics)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # do the same thing as on_train_batch_end

        if not self.train_progress_bar.disable:
            metrics = self.get_metrics(trainer, pl_module)
            metrics = self.filter_metrics(metrics)

            metrics = self.add_time(metrics)
            metrics = self.add_learning_rate(metrics, trainer)

            metrics = self.sort_metrics(metrics)

            self.train_progress_bar.set_postfix(metrics)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            metrics = self.get_metrics(trainer, pl_module)
            metrics = self.filter_metrics(metrics)

            metrics = self.add_time(metrics)
            metrics = self.add_learning_rate(metrics, trainer)

            metrics = self.sort_metrics(metrics)

            _update_n(self.val_progress_bar, n)
            self.val_progress_bar.set_postfix(metrics)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._train_progress_bar is not None and trainer.state.fn == "fit":
            metrics = self.get_metrics(trainer, pl_module)
            metrics = self.filter_metrics(metrics)

            metrics = self.add_time(metrics)
            metrics = self.add_learning_rate(metrics, trainer)

            metrics = self.sort_metrics(metrics)

            self.val_progress_bar.set_postfix(metrics)
        self.val_progress_bar.close()
        self.reset_dataloader_idx_tracker()
