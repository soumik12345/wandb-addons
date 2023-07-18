import os
import sys
from typing import Dict, List, Optional, Union

import keras_core as keras
from keras_core.callbacks import ModelCheckpoint
from wandb.sdk.lib import telemetry
from wandb.sdk.lib.paths import StrPath

import wandb

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


Mode = Literal["auto", "min", "max"]
SaveStrategy = Literal["epoch"]


def _log_artifact(
    filepath: StrPath, aliases: Optional[List] = None, metadata: Optional[Dict] = None
):
    aliases = ["latest"] if aliases is None else aliases + ["latest"]
    metadata = wandb.run.config.as_dict() if metadata is None else metadata
    model_checkpoint_artifact = wandb.Artifact(
        f"run_{wandb.run.id}_model", type="model", metadata=metadata
    )
    if os.path.isfile(filepath):
        model_checkpoint_artifact.add_file(filepath)
    elif os.path.isdir(filepath):
        model_checkpoint_artifact.add_dir(filepath)
    else:
        raise FileNotFoundError(f"No such file or directory {filepath}")
    wandb.log_artifact(model_checkpoint_artifact, aliases=aliases or [])


class WandbModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        filepath: StrPath,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: Mode = "auto",
        save_freq: Union[SaveStrategy, int] = "epoch",
        initial_value_threshold: Optional[float] = None,
    ):
        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before `WandbModelCheckpoint()`"
            )
        with telemetry.context(run=wandb.run) as tel:
            tel.feature.keras_model_checkpoint = True

        super().__init__(
            filepath,
            monitor,
            verbose,
            save_best_only,
            save_weights_only,
            mode,
            save_freq,
            initial_value_threshold,
        )

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        if self._should_save_on_batch(batch):
            _log_artifact(self.filepath, aliases=[f"batch_{batch}"])

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.save_freq == "epoch":
            _log_artifact(self.filepath, aliases=[f"epoch_{epoch}"])
