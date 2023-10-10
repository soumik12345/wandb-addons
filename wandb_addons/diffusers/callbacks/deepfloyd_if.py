from typing import Dict, List, Optional, Union

import torch
from diffusers import DiffusionPipeline

from .base import BaseDiffusersCallback


class IFCallback(BaseDiffusersCallback):
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        prompt: Union[str, List[str]],
        wandb_project: str,
        wandb_entity: Optional[str] = None,
        weave_mode: bool = False,
        num_inference_steps: int = 100,
        num_images_per_prompt: Optional[int] = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        configs: Optional[Dict] = None,
        **kwargs
    ) -> None:
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
            **kwargs
        )
        self.starting_step = 0
        self.log_step = self.num_inference_steps - 1

    def generate(self, latents: torch.FloatTensor) -> List:
        images = (latents / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images, nsfw_detected, watermark_detected = self.pipeline.run_safety_checker(
            images, self.pipeline._execution_device, latents.dtype
        )
        images = self.pipeline.numpy_to_pil(images)
        return images
