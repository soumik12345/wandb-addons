import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Union

import wandb

import torch

from ignite.engine import Engine
from ignite.distributed import idist
from ignite.handlers.checkpoint import BaseSaveHandler


class WandbModelCheckpointSaver(BaseSaveHandler):
    @idist.one_rank_only()
    def __init__(self):
        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before `WandbModelCheckpointSaver()`"
            )

        self.checkpoint_dir = tempfile.mkdtemp()

    @idist.one_rank_only()
    def __call__(self, checkpoint: Mapping, filename: Union[str, Path]):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)

        artifact = wandb.Artifact(f"{wandb.run}-checkpoint", type="model")

        if os.path.isfile(checkpoint_path):
            artifact.add_file(checkpoint_path)
        elif os.path.isdir(checkpoint_path):
            artifact.add_dir(checkpoint_path)
        else:
            raise wandb.Error(
                f"Unable to local checkpoint path {checkpoint_path} to artifact"
            )

        wandb.log_artifact(artifact)

    @idist.one_rank_only()
    def remove(self, filename):
        shutil.rmtree(filename)
