from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

import torch
import wandb

from monai.handlers import TensorBoardStatsHandler

from monai.config import IgniteInfo
from monai.utils import optional_import, min_version, is_scalar

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


class WandbStatsHandler(TensorBoardStatsHandler):
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
        state_attributes: Optional[Sequence[str]] = None,
        tag_name: str = DEFAULT_TAG,
    ):
        super().__init__(
            summary_writer=None,
            log_dir=None,
            iteration_log=iteration_log,
            epoch_log=epoch_log,
            epoch_event_writer=epoch_event_writer,
            epoch_interval=epoch_interval,
            iteration_event_writer=iteration_event_writer,
            iteration_interval=iteration_interval,
            output_transform=output_transform,
            global_epoch_transform=global_epoch_transform,
            state_attributes=state_attributes,
            tag_name=tag_name,
        )

    def _default_epoch_writer(self, engine: Engine, writer) -> None:
        current_epoch = self.global_epoch_transform(engine.state.epoch)
        summary_dict = engine.state.metrics

        for key, value in summary_dict.items():
            if is_scalar(value):
                value = value.item() if isinstance(value, torch.Tensor) else value
                wandb.log({key: value})

        if self.state_attributes is not None:
            for attr in self.state_attributes:
                value = getattr(engine.state, attr, None)
                value = value.item() if isinstance(value, torch.Tensor) else value
                wandb.log({attr: value})

    def _default_iteration_writer(self, engine: Engine, writer) -> None:
        loss = self.output_transform(engine.state.output)
        if loss is None:
            return  # do nothing if output is empty
        log_dict = dict()
        if isinstance(loss, dict):
            for key, value in loss.items():
                if not is_scalar(value):
                    warnings.warn(
                        "ignoring non-scalar output in WandbStatsHandler,"
                        " make sure `output_transform(engine.state.output)` returns"
                        " a scalar or dictionary of key and scalar pairs to avoid this warning."
                        " {}:{}".format(name, type(value))
                    )
                    continue  # not plot multi dimensional output
                log_dict[key] = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
        elif is_scalar(loss):  # not printing multi dimensional output
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
