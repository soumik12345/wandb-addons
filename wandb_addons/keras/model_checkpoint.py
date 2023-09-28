import os
import sys
from typing import Dict, List, Optional, Union

import keras
import wandb
from keras.callbacks import ModelCheckpoint
from wandb.sdk.lib import telemetry
from wandb.sdk.lib.paths import StrPath

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
    """`WandbModelCheckpoint` automatically logs model checkpoints to W&B and versions
    them as [W&B artifacts](https://docs.wandb.ai/guides/artifacts).

    Since this callback is subclassed from
    [`keras.callbacks.ModelCheckpoint`](https://keras.io/keras_core/api/callbacks/model_checkpoint/),
    the checkpointing logic is taken care of by the parent callback.

    This callback is to be used in conjunction with training using `model.fit()` to save
    a model or weights (in a checkpoint file) at some interval. The model checkpoints
    will be logged as W&B Artifacts. You can learn more here:
    https://docs.wandb.ai/guides/artifacts

    This callback provides the following features:
        - Save the model that has achieved "best performance" based on "monitor".
        - Save the model at the end of every epoch regardless of the performance.
        - Save the model at the end of epoch or after a fixed number of training
            batches.
        - Save only model weights, or save the whole model.
        - Save the model either in SavedModel format or in `.h5` format.

    !!! example "Example notebooks:"
        - [Image Classification using Keras Core](../examples/image_classification).

    Arguments:
        filepath (wandb.sdk.lib.paths.StrPath): path to save the model file. `filepath`
            can contain named formatting options, which will be filled by the value
            of `epoch` and keys in `logs` (passed in `on_epoch_end`). For example:
            if `filepath` is `model-{epoch:02d}-{val_loss:.2f}`, then the
            model checkpoints will be saved with the epoch number and the
            validation loss in the filename.
        monitor (str): The metric name to monitor. Default to "val_loss".
        verbose (int): Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
            displays messages when the callback takes an action.
        save_best_only (bool): if `save_best_only=True`, it only saves when the model
            is considered the "best" and the latest best model according to the
            quantity monitored will not be overwritten. If `filepath` doesn't contain
            formatting options like `{epoch}` then `filepath` will be overwritten by
            each new better model locally. The model logged as an artifact will still
            be associated with the correct `monitor`.  Artifacts will be uploaded
            continuously and versioned separately as a new best model is found.
        save_weights_only (bool): if True, then only the model's weights will be saved.
        mode (Literal["auto", "min", "max"]): one of {'auto', 'min', 'max'}. For
            `val_acc`, this should be `max`, for `val_loss` this should be `min`, etc.
        save_freq (Union[SaveStrategy, int]): `epoch` or integer. When using `'epoch'`,
            the callback saves the model after each epoch. When using an integer, the
            callback saves the model at end of this many batches.
            Note that when monitoring validation metrics such as `val_acc` or
            `val_loss`, save_freq must be set to "epoch" as those metrics are only
            available at the end of an epoch.
        initial_value_threshold (Optional[float]): Floating point initial "best" value
            of the metric to be monitored.
    """

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
