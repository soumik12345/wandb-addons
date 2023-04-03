from typing import Optional, Callable

import ciclo
from ciclo.logging import Logs
from ciclo.types import Batch, S
from ciclo.timetracking import Elapsed
from ciclo.loops.loop import LoopCallbackBase
from ciclo.callbacks import LoopState, CallbackOutput

import wandb


class WandBLogger(LoopCallbackBase[S]):
    def __init__(self, additional_logging: Optional[Callable] = None):
        self.additional_logging = additional_logging

    def __call__(self, elapsed: Elapsed, state: S, logs: Optional[Logs] = None):
        wandb.log(logs)

    def __loop_callback__(self, loop_state: LoopState[S]) -> CallbackOutput[S]:
        self(loop_state.elapsed, loop_state.state, loop_state.accumulated_logs)
        return Logs(), loop_state.state

    def on_epoch_end(
        self, state, batch, elapsed, loop_state: LoopState[S]
    ) -> CallbackOutput[S]:
        return self.__loop_callback__(loop_state)
