from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import wandb

from ignite.engine import Engine
from ignite.handlers import ModelCheckpoint


class WandbModelCheckpointHandler(ModelCheckpoint):
    def __init__(
        self,
        dirname: Union[str, Path],
        filename_prefix: str = "",
        save_interval: Optional[int] = None,
        score_function: Optional[Callable] = None,
        score_name: Optional[str] = None,
        n_saved: Union[int, None] = 1,
        atomic: bool = True,
        require_empty: bool = True,
        create_dir: bool = True,
        save_as_state_dict: bool = True,
        global_step_transform: Optional[Callable] = None,
        archived: bool = False,
        filename_pattern: Optional[str] = None,
        include_self: bool = False,
        greater_or_equal: bool = False,
        save_on_rank: int = 0,
        **kwargs: Any,
    ):
        super().__init__(
            dirname,
            filename_prefix,
            save_interval,
            score_function,
            score_name,
            n_saved,
            atomic,
            require_empty,
            create_dir,
            save_as_state_dict,
            global_step_transform,
            archived,
            filename_pattern,
            include_self,
            greater_or_equal,
            save_on_rank,
            **kwargs,
        )

    def __call__(self, engine: Engine, to_save: Mapping):
        super().__call__(engine, to_save)
        artifact = wandb.Artifact(f"run-{wandb.run.id}", type="model")
        artifact.add_file(self.last_checkpoint)
        wandb.log_artifact(artifact)
