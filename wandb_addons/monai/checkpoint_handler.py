import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Union

import wandb

import torch

from ignite.engine import Engine
from ignite.distributed import one_rank_only
from ignite.handlers.checkpoint import BaseSaveHandler


class WandbModelCheckpointSaver(BaseSaveHandler):
    """`WandbModelCheckpointSaver` is a save handler for PyTorch Ignite that saves model checkpoints as
    [Weights & Biases Artifacts](https://docs.wandb.ai/guides/artifacts).

    Usage:

    ```python
    from wandb_addons.monai import WandbModelCheckpointSaver

    checkpoint_handler = Checkpoint(
        {"model": model, "optimizer": optimizer},
        WandbModelCheckpointSaver(),
        n_saved=1,
        filename_prefix="best_checkpoint",
        score_name=metric_name,
        global_step_transform=global_step_from_engine(trainer)
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)
    ```
    """

    @one_rank_only()
    def __init__(self):
        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before `WandbModelCheckpointSaver()`"
            )

        self.checkpoint_dir = tempfile.mkdtemp()

    @one_rank_only()
    def __call__(self, checkpoint: Mapping, filename: Union[str, Path]):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)

        artifact = wandb.Artifact(f"{wandb.run.id}-checkpoint", type="model")

        if os.path.isfile(checkpoint_path):
            artifact.add_file(checkpoint_path)
        elif os.path.isdir(checkpoint_path):
            artifact.add_dir(checkpoint_path)
        else:
            raise wandb.Error(
                f"Unable to local checkpoint path {checkpoint_path} to artifact"
            )

        wandb.log_artifact(artifact)

    @one_rank_only()
    def remove(self, filename):
        if os.path.exists(filename):
            shutil.rmtree(filename)
