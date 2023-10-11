from typing import Any, Dict, List, Optional, Union

import wandb
from diffusers import DiffusionPipeline

from .base_callback import BaseDiffusersCallback
from ..utils import chunkify


class BaseMultiPipelineCallback(BaseDiffusersCallback):
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        prompt: Union[str, List[str]],
        wandb_project: str,
        wandb_entity: Optional[str] = None,
        weave_mode: bool = False,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        configs: Optional[Dict] = None,
        initial_stage_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.stage_name = (
            initial_stage_name if initial_stage_name is not None else "stage_1"
        )
        self.stage_counter = 1
        super().__init__(
            pipeline,
            prompt,
            wandb_project,
            wandb_entity,
            weave_mode,
            num_inference_steps,
            num_images_per_prompt,
            negative_prompt,
            configs,
            **kwargs,
        )

    def update_configs(self) -> None:
        additional_configs = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_images_per_prompt": self.num_images_per_prompt,
            self.stage_name: {
                "pipeline": dict(self.pipeline.config),
                "num_inference_steps": self.num_inference_steps,
            },
        }
        self.configs = (
            {**self.configs, **additional_configs}
            if self.configs is not None
            else additional_configs
        )

    def add_stage(
        self,
        pipeline: DiffusionPipeline,
        num_inference_steps: Optional[int] = None,
        stage_name: Optional[str] = None,
        configs: Optional[Dict] = None,
    ) -> None:
        self.pipeline = pipeline
        self.num_inference_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else self.num_inference_steps
        )
        self.stage_counter += 1
        self.stage_name = (
            stage_name if stage_name is not None else f"stage_{self.stage_counter}"
        )
        additional_configs = {
            self.stage_name: {
                "pipeline": dict(self.pipeline.config),
                "num_inference_steps": self.num_inference_steps,
                "stage-sequence": self.stage_counter,
            }
        }
        if configs is not None:
            additional_configs[self.stage_name].update(configs)
        self.configs.update(additional_configs)
        if wandb.run is not None:
            wandb.config.update(additional_configs)

    def at_initial_step(self):
        if self.stage_counter == 1:
            super().at_initial_step()

    def build_wandb_table(self) -> None:
        super().build_wandb_table()
        self.table_columns = ["Stage-Sequence", "Stage-Name"] + self.table_columns

    def populate_table_row(self, prompt: str, negative_prompt: str, image: Any) -> None:
        super().populate_table_row(prompt, negative_prompt, image)
        if self.weave_mode:
            self.table_row.update(
                {"Stage-Sequence": self.stage_counter, "Stage-Name": self.stage_name}
            )
        else:
            self.table_row = [self.stage_counter, self.stage_name] + self.table_row
