from typing import Dict, List, Optional, Union

import PIL
import torch
from diffusers import StableDiffusionPipeline

import wandb

from .base import BaseDiffusersBaseCallback


class StableDiffusionCallback(BaseDiffusersBaseCallback):
    def __init__(
        self,
        pipeline: StableDiffusionPipeline,
        prompt: str | List[str],
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        num_images_per_prompt: int | None = 1,
        negative_prompt: str | List[str] | None = None,
        configs: Dict | None = None,
        **kwargs
    ) -> None:
        super().__init__(
            pipeline,
            prompt,
            wandb_project,
            wandb_entity,
            num_inference_steps,
            num_images_per_prompt,
            negative_prompt,
            configs,
            **kwargs
        )
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1.0
        wandb.config.update(
            {
                "guidance_scale": self.guidance_scale,
                "do_classifier_free_guidance": do_classifier_free_guidance,
            }
        )

    def build_wandb_table(self) -> None:
        super().build_wandb_table()
        self.wandb_table.columns += [
            "Guidance Scale",
            "Do-Classifier-Free-Guidance",
        ]

    def generate(self, latents: torch.FloatTensor) -> PIL.Image:
        text_embeddings = self.pipeline._encode_prompt(
            self.prompt,
            self.pipeline._execution_device,
            self.num_images_per_prompt,
            self.do_classifier_free_guidance,
            self.negative_prompt,
        )
        images = self.pipeline.decode_latents(latents)
        images, _ = self.pipeline.run_safety_checker(
            images, self.pipeline._execution_device, text_embeddings.dtype
        )
        images = self.pipeline.numpy_to_pil(images)
        return images

    def add_data_to_wandb_table(
        self, prompt: str, negative_prompt: str, image: PIL.Image, *args
    ) -> None:
        self.wandb_table.add_data(
            prompt,
            negative_prompt,
            wandb.Image(image),
            self.guidance_scale,
            self.do_classifier_free_guidance,
        )
