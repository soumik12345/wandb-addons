from typing import Dict, List, Optional, Union

import torch
from diffusers import IFPipeline, IFSuperResolutionPipeline

from .base import BaseMultiPipelineCallback


class IFCallback(BaseMultiPipelineCallback):
    def __init__(
        self,
        pipeline: Union[IFPipeline, IFSuperResolutionPipeline],
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

    def add_stage(
        self,
        pipeline: Union[IFPipeline, IFSuperResolutionPipeline],
        num_inference_steps: Optional[int] = None,
        stage_name: Optional[str] = None,
    ) -> None:
        assert isinstance(pipeline, IFPipeline) or isinstance(
            pipeline, IFSuperResolutionPipeline
        ), "IFCallback only supports IFPipeline and IFSuperResolutionPipeline"
        super().add_stage(pipeline, num_inference_steps, stage_name)
        if isinstance(pipeline, IFSuperResolutionPipeline):
            self.starting_step = 0
            self.log_step = self.starting_step + self.num_inference_steps - 1
