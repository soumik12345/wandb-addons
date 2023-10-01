from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import PIL
import torch
from diffusers import DiffusionPipeline

import wandb

from .utils import chunkify


class BaseDiffusersBaseCallback(ABC):
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        prompt: Union[str, List[str]],
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        configs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.prompt = prompt
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.num_inference_steps = num_inference_steps
        self.num_images_per_prompt = num_images_per_prompt
        self.negative_prompt = negative_prompt
        self.configs = configs
        self.wandb_table = None
        self.initialize_wandb(wandb_project, wandb_entity)
        self.build_wandb_table()

    def update_configs(self) -> None:
        additional_configs = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "num_images_per_prompt": self.num_images_per_prompt,
            "pipeline": dict(self.pipeline.config),
        }
        self.configs = (
            {**self.configs, **additional_configs}
            if self.configs is not None
            else additional_configs
        )

    def initialize_wandb(self, wandb_project, wandb_entity) -> None:
        if wandb.run is None:
            if wandb_project is not None:
                self.update_configs()
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    job_type="text-to-image",
                )
            else:
                wandb.termerror("The parameter wandb_project must be provided.")

    def build_wandb_table(self) -> None:
        self.wandb_table = wandb.Table(
            columns=[
                "Prompt",
                "Negative-Prompt",
                "Generated-Image",
                "Inference-Step",
            ]
        )

    @abstractmethod
    def generate(self, latents: torch.FloatTensor) -> PIL.Image:
        pass

    @abstractmethod
    def add_data_to_wandb_table(
        self, prompt: str, negative_prompt: str, image: PIL.Image, *args
    ) -> None:
        pass

    def __call__(self, step: int, timestep: int, latents: torch.FloatTensor):
        if step % self.num_inference_steps == 0:
            images = self.generate(latents)
            prompt_logging = (
                self.prompt if isinstance(self.prompt, list) else [self.prompt]
            )
            negative_prompt_logging = (
                self.negative_prompt
                if isinstance(self.negative_prompt, list)
                else [self.negative_prompt] * len(prompt_logging)
            )
            images = chunkify(images, len(prompt_logging))
            for idx in range(len(prompt_logging)):
                for image in images[idx]:
                    self.add_data_to_wandb_table(
                        prompt_logging[idx], negative_prompt_logging[idx], image
                    )
            wandb.log({"Generated-Images": self.wandb_table})
            wandb.finish()
