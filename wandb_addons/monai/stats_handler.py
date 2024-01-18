import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import torch
import wandb
from monai.config import IgniteInfo
from monai.utils import is_scalar, min_version, optional_import

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

DEFAULT_TAG = "Loss"


class WandbStatsHandler:
    """
    `WandbStatsHandler` defines a set of Ignite Event-handlers for all the Weights & Biases logging
    logic. It can be used for any Ignite Engine(trainer, validator and evaluator) and support both
    epoch level and iteration level. The expected data source is Ignite `engine.state.output` and
    `engine.state.metrics`.

    Default behaviors:
        - When EPOCH_COMPLETED, write each dictionary item in `engine.state.metrics` to
            Weights & Biases.
        - When ITERATION_COMPLETED, write each dictionary item in
            `self.output_transform(engine.state.output)` to Weights & Biases.

    **Usage:**

    ```python
    # WandbStatsHandler for logging training metrics and losses at
    # every iteration to Weights & Biases
    train_wandb_stats_handler = WandbStatsHandler(output_transform=lambda x: x)
    train_wandb_stats_handler.attach(trainer)

    # WandbStatsHandler for logging validation metrics and losses at
    # every iteration to Weights & Biases
    val_wandb_stats_handler = WandbStatsHandler(
        output_transform=lambda x: None,
        global_epoch_transform=lambda x: trainer.state.epoch,
    )
    val_wandb_stats_handler.attach(evaluator)
    ```

    ??? example "Example notebooks:"
        - [3D classification using MonAI](../examples/densenet_training_dict).
        - [3D segmentation using MonAI](../examples/unet_3d_segmentation).

    ??? note "Pull Request to add `WandbStatsHandler` to MonAI repository"

        There is an [open pull request](https://github.com/Project-MONAI/MONAI/pull/6305)
        to add `WandbStatsHandler` to [MonAI](https://github.com/Project-MONAI/MONAI).


    Args:
        iteration_log (bool): Whether to write data to Weights & Biases when iteration completed,
            default to `True`.
        epoch_log (bool): Whether to write data to Weights & Biases when epoch completed, default to
            `True`.
        epoch_event_writer (Optional[Callable[[Engine, Any], Any]]): Customized callable
            Weights & Biases writer for epoch level. Must accept parameter "engine" and
            "summary_writer", use default event writer if None.
        epoch_interval (int): The epoch interval at which the epoch_event_writer is called. Defaults
            to 1.
        iteration_event_writer (Optional[Callable[[Engine, Any], Any]]): Customized callable
            Weights & Biases writer for iteration level. Must accept parameter "engine" and
            "summary_writer", use default event writer if None.
        iteration_interval (int): The iteration interval at which the iteration_event_writer is
            called. Defaults to 1.
        output_transform (Callable): A callable that is used to transform the
            `ignite.engine.state.output` into a scalar to plot, or a dictionary of `{key: scalar}`. In
            the latter case, the output string will be formatted as key: value. By default this value
            plotting happens when every iteration completed. The default behavior is to print loss
            from output[0] as output is a decollated list and we replicated loss value for every item
            of the decollated list. `engine.state` and `output_transform` inherit from the ignite
            concept: https://pytorch.org/ignite/concepts.html#state, explanation and usage example are
            in the tutorial:
            https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
        global_epoch_transform (Callable): A callable that is used to customize global epoch number. For
            example, in evaluation, the evaluator engine might want to use trainer engines epoch number
            when plotting epoch vs metric curves.
        state_attributes (Optional[Sequence[str]]): Expected attributes from `engine.state`, if provided,
            will extract them when epoch completed.
        tag_name (str): When iteration output is a scalar, tag_name is used to plot, defaults to `'Loss'`.
    """

    ITERATION_KEY = "iteration"
    EPOCH_KEY = "epoch"

    def __init__(
        self,
        iteration_log: bool = True,
        epoch_log: bool = True,
        epoch_event_writer: Optional[Callable[[Engine, Any], Any]] = None,
        epoch_interval: int = 1,
        iteration_event_writer: Optional[Callable[[Engine, Any], Any]] = None,
        iteration_interval: int = 1,
        output_transform: Callable = lambda x: x[0],
        global_epoch_transform: Callable = lambda x: x,
        global_iter_transform: Callable = lambda x: x,
        state_attributes: Optional[Sequence[str]] = None,
        tag_name: str = DEFAULT_TAG,
    ):
        if wandb.run is None:
            raise wandb.Error("You must call `wandb.init()` before WandbStatsHandler()")

        self.iteration_log = iteration_log
        self.epoch_log = epoch_log
        self.epoch_event_writer = epoch_event_writer
        self.epoch_interval = epoch_interval
        self.iteration_event_writer = iteration_event_writer
        self.iteration_interval = iteration_interval
        self.output_transform = output_transform
        self.global_epoch_transform = global_epoch_transform
        self.global_iter_transform = global_iter_transform
        self.state_attributes = state_attributes
        self.tag_name = tag_name

        self.metrics_history = set()

        # Necessary first step for setting default x-axes on wandb UI
        wandb.define_metric(self.EPOCH_KEY)
        self.metrics_history.add(self.EPOCH_KEY)

        wandb.define_metric(self.ITERATION_KEY)
        self.metrics_history.add(self.ITERATION_KEY)

    def attach(self, engine: Engine) -> None:
        """
        Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine (ignite.engine.engine.Engine): Ignite Engine, it can be a trainer, validator
                or evaluator.
        """
        if self.iteration_log and not engine.has_event_handler(
            self.iteration_completed, Events.ITERATION_COMPLETED
        ):
            engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=self.iteration_interval),
                self.iteration_completed,
            )
        if self.epoch_log and not engine.has_event_handler(
            self.epoch_completed, Events.EPOCH_COMPLETED
        ):
            engine.add_event_handler(
                Events.EPOCH_COMPLETED(every=self.epoch_interval), self.epoch_completed
            )

    def epoch_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation epoch completed Event. Write epoch level events
        to Weights & Biases, default values are from Ignite `engine.state.metrics` dict.

        Args:
            engine (ignite.engine.engine.Engine): Ignite Engine, it can be a trainer, validator
                or evaluator.
        """
        if self.epoch_event_writer is not None:
            self.epoch_event_writer(engine)
        else:
            self._default_epoch_writer(engine)

    def iteration_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation iteration completed Event. Write iteration level
        events to Weighs & Biases, default values are from Ignite `engine.state.output`.

        Args:
            engine (ignite.engine.engine.Engine): Ignite Engine, it can be a trainer, validator
                or evaluator.
        """
        if self.iteration_event_writer is not None:
            self.iteration_event_writer(engine)
        else:
            self._default_iteration_writer(engine)

    def _define_metric_if_new(self, key, step_metric):
        if key not in self.metrics_history:
            wandb.define_metric(key, step_metric=step_metric)
            self.metrics_history.add(key)


    def _default_epoch_writer(self, engine: Engine) -> None:
        """
        Execute epoch level event write operation. Default to write the values from Ignite
        `engine.state.metrics` dict and write the values of specified attributes of `engine.state`
        to [Weights & Biases](https://wandb.ai/site).

        Args:
            engine (ignite.engine.engine.Engine): Ignite Engine, it can be a trainer, validator
                or evaluator.
        """

        epoch_step_dict = dict()
        epoch_step_dict = {self.EPOCH_KEY: self.global_epoch_transform(engine.state.epoch)}

        summary_dict = engine.state.metrics
        for key, value in summary_dict.items():
            if is_scalar(value):
                value = value.item() if isinstance(value, torch.Tensor) else value
                self._define_metric_if_new(key, self.EPOCH_KEY)
                log_dict = {key: value}
                log_dict.update(epoch_step_dict)
                wandb.log(log_dict)

        if self.state_attributes is not None:
            for attr in self.state_attributes:
                value = getattr(engine.state, attr, None)
                value = value.item() if isinstance(value, torch.Tensor) else value
                self._define_metric_if_new(attr, self.EPOCH_KEY)
                log_dict = {attr: value}
                log_dict.update(epoch_step_dict)
                wandb.log(log_dict)

    def _default_iteration_writer(self, engine: Engine) -> None:
        """
        Execute iteration level event write operation based on Ignite `engine.state.output` data.
        Extract the values from `self.output_transform(engine.state.output)`. Since
        `engine.state.output` is a decollated list and we replicated the loss value for every item
        of the decollated list, the default behavior is to track the loss from `output[0]`.

        Args:
            engine (ignite.engine.engine.Engine): Ignite Engine, it can be a trainer, validator
                or evaluator.
        """
        loss = self.output_transform(engine.state.output)
        if loss is None:
            return  # do nothing if output is empty

        log_dict = {self.ITERATION_KEY: self.global_iter_transform(engine.state.iteration)}
        if isinstance(loss, dict):
            for key, value in loss.items():
                if not is_scalar(value):
                    warnings.warn(
                        "ignoring non-scalar output in WandbStatsHandler,"
                        " make sure `output_transform(engine.state.output)` returns"
                        " a scalar or dictionary of key and scalar pairs to avoid this warning."
                        " {}:{}".format(key, type(value))
                    )
                    continue  # not plot multi dimensional output
                self._define_metric_if_new(key, self.ITERATION_KEY)
                log_dict[key] = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
        elif is_scalar(loss):  # not printing multi dimensional output
            self._define_metric_if_new(self.tag_name, self.ITERATION_KEY)
            log_dict[self.tag_name] = (
                loss.item() if isinstance(loss, torch.Tensor) else loss
            )
        else:
            warnings.warn(
                "ignoring non-scalar output in WandbStatsHandler,"
                " make sure `output_transform(engine.state.output)` returns"
                " a scalar or a dictionary of key and scalar pairs to avoid this warning."
                " {}".format(type(loss))
            )

        wandb.log(log_dict)

    def close(self):
        """Close `WandbStatsHandler`"""
        wandb.finish()
