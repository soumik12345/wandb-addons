from typing import Callable, TYPE_CHECKING

import torch
import wandb
import numpy as np

from monai.config import IgniteInfo
from monai.utils import optional_import, min_version

from .utils import plot_2d_or_3d_image

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


class WandBImageHandler:
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
    ) -> None:
        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before `WandbStatsHandler()`"
            )

        self.interval = interval
        self.epoch_level = epoch_level
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.global_iter_transform = global_iter_transform
        self.index = index
        self.frame_dim = frame_dim
        self.max_frames = max_frames
        self.max_channels = max_channels

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

    def __call__(self, engine: Engine):
        step = self.global_iter_transform(
            engine.state.epoch if self.epoch_level else engine.state.iteration
        )

        show_images = self.batch_transform(engine.state.batch)[0][self.index]
        if isinstance(show_images, torch.Tensor):
            show_images = show_images.detach().cpu().numpy()

        show_labels = self.batch_transform(engine.state.batch)[1][self.index]
        if isinstance(show_labels, torch.Tensor):
            show_labels = show_labels.detach().cpu().numpy()

        show_outputs = self.output_transform(engine.state.output)[self.index]
        if isinstance(show_outputs, torch.Tensor):
            show_outputs = show_outputs.detach().cpu().numpy()

        if show_images is not None:
            if not isinstance(show_images, np.ndarray):
                raise TypeError(
                    "output_transform(engine.state.output)[0] must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_images).__name__}."
                )
            plot_2d_or_3d_image(
                image_key=None,
                # add batch dim and plot the first item
                data=show_images[None],
                index=0,
                max_channels=self.max_channels,
                frame_dim=self.frame_dim,
                max_frames=self.max_frames,
            )

        if show_labels is not None:
            if not isinstance(show_labels, np.ndarray):
                raise TypeError(
                    "batch_transform(engine.state.batch)[1] must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_labels).__name__}."
                )
            plot_2d_or_3d_image(
                image_key=None,
                data=show_labels[None],
                index=0,
                max_channels=self.max_channels,
                frame_dim=self.frame_dim,
                max_frames=self.max_frames,
            )

        if show_outputs is not None:
            if not isinstance(show_outputs, np.ndarray):
                raise TypeError(
                    "output_transform(engine.state.output) must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_outputs).__name__}."
                )
            plot_2d_or_3d_image(
                image_key=None,
                data=show_outputs[None],
                index=0,
                max_channels=self.max_channels,
                frame_dim=self.frame_dim,
                max_frames=self.max_frames,
            )
