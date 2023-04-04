from typing import Optional, Callable

import ciclo
from ciclo.logging import Logs
from ciclo.types import Batch, S
from ciclo.timetracking import Elapsed
from ciclo.loops.loop import LoopCallbackBase
from ciclo.callbacks import LoopState, CallbackOutput

import wandb

from ..utils import flatten_nested_dictionaries


class wandb_logging(LoopCallbackBase[S]):
    """A [ciclo](https://github.com/cgarciae/ciclo) callback for logging to Weights & Biases."""

    def __init__(self, additional_logging: Optional[Callable] = None):
        """
        # Arguments:
            additional_logging: Optional[Callable].
                A function to be called after each logging step and can be used to log
                additional values/media to Weights & Biases.
        """
        self.additional_logging = additional_logging

    def __call__(self, elapsed: Elapsed, state: S, logs: Optional[Logs] = None):
        wandb.log(flatten_nested_dictionaries(logs, sep="/"))
        if self.additional_logging is not None:
            self.additional_logging()

    def __loop_callback__(self, loop_state: LoopState[S]) -> CallbackOutput[S]:
        self(loop_state.elapsed, loop_state.state, loop_state.accumulated_logs)
        return Logs(), loop_state.state

    def on_epoch_end(
        self, state, batch, elapsed, loop_state: LoopState[S]
    ) -> CallbackOutput[S]:
        return self.__loop_callback__(loop_state)
