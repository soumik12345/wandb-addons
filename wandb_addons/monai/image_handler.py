from typing import Callable, TYPE_CHECKING

import torch
import wandb
import numpy as np

from monai.config import IgniteInfo
from monai.utils import optional_import, min_version

Events, _ = optional_import(
    "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events"
)

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine",
        IgniteInfo.OPT_IMPORT_VERSION,
        min_version,
        "Engine",
        as_type="decorator",
    )


class WandbImageHandler:
    def __init__(
        self,
        interval: int = 1,
        epoch_level: bool = True,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        global_iter_transform: Callable = lambda x: x,
        index: int = 0,
        max_channels: int = 1,
        frame_dim: int = -3,
        max_frames: int = 64,
    ):
        self.interval = interval
        self.epoch_level = epoch_level
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.global_iter_transform = global_iter_transform
        self.index = index
        self.max_channels = max_channels
        self.frame_dim = frame_dim
        self.max_frames = max_frames

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self)
        else:
            engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=self.interval), self
            )

    def __call__(self, engine: Engine) -> None:
        step = self.global_iter_transform(
            engine.state.epoch if self.epoch_level else engine.state.iteration
        )
        show_images = self.batch_transform(engine.state.batch)[0][self.index]
        if isinstance(show_images, torch.Tensor):
            show_images = show_images.detach().cpu().numpy()
        if show_images is not None:
            if not isinstance(show_images, np.ndarray):
                raise TypeError(
                    "output_transform(engine.state.output)[0] must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_images).__name__}."
                )
            # Added temporarily for testing
            print(show_images.shape)
