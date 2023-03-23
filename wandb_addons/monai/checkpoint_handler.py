from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import wandb

from ignite.engine import Engine
from ignite.handlers import ModelCheckpoint


class WandbModelCheckpointHandler(ModelCheckpoint):
    """WandbModelCheckpointHandler  inherits from :class:`~ignite.handlers.ModelCheckpoint`, can be used
    to periodically save objects as [Weights & Biases artifacts](https://docs.wandb.ai/guides/artifacts).

    This handler expects two arguments:

        - an :class:`~ignite.engine.engine.Engine` object
        - a `dict` mapping names (`str`) to objects that should be saved to disk.
    """

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
        """Args:
        dirname: Directory path where objects will be saved.
        filename_prefix: Prefix for the file names to which objects will be saved. See Notes of
            :class:`~ignite.handlers.checkpoint.Checkpoint` for more details.
        score_function: if not None, it should be a function taking a single argument, an
            :class:`~ignite.engine.engine.Engine` object, and return a score (`float`). Objects with highest scores
            will be retained.
        score_name: if ``score_function`` not None, it is possible to store its value using
            `score_name`. See Examples of :class:`~ignite.handlers.checkpoint.Checkpoint` for more details.
        n_saved: Number of objects that should be kept on disk. Older files will be removed. If set to
            `None`, all objects are kept.
        atomic: If True, objects are serialized to a temporary file, and then moved to final
            destination, so that files are guaranteed to not be damaged (for example if exception
            occurs during saving).
        require_empty: If True, will raise exception if there are any files starting with
            ``filename_prefix`` in the directory ``dirname``.
        create_dir: If True, will create directory ``dirname`` if it does not exist.
        global_step_transform: global step transform function to output a desired global step.
            Input of the function is `(engine, event_name)`. Output of function should be an integer.
            Default is None, global_step based on attached engine. If provided, uses function output as global_step.
            To setup global step from another engine, please use :meth:`~ignite.handlers.global_step_from_engine`.
        archived: Deprecated argument as models saved by `torch.save` are already compressed.
        filename_pattern: If ``filename_pattern`` is provided, this pattern will be used to render
            checkpoint filenames. If the pattern is not defined, the default pattern would be used.
            See :class:`~ignite.handlers.checkpoint.Checkpoint` for details.
        include_self: Whether to include the `state_dict` of this object in the checkpoint. If `True`, then
            there must not be another object in ``to_save`` with key ``checkpointer``.
        greater_or_equal: if `True`, the latest equally scored model is stored. Otherwise, the first model.
            Default, `False`.
        save_on_rank: Which rank to save the objects on, in the distributed configuration. Used to
            instantiate a :class:`~ignite.handlers.DiskSaver` and is also passed to the parent class.
        kwargs: Accepted keyword arguments for `torch.save` or `xm.save` in `DiskSaver`.
        """
        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before WandbModelCheckpointHandler()"
            )

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
