from typing import Dict, List, Optional, Union

import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

from ..base import BaseMultiPipelineCallback


class StableDiffusionXLCallback(BaseMultiPipelineCallback):
    def __init__(
        self,
        pipeline: Union[DiffusionPipeline, StableDiffusionXLPipeline],
        prompt: Union[str, List[str]],
        wandb_project: str,
        wandb_entity: Optional[str] = None,
        weave_mode: bool = False,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        configs: Optional[Dict] = None,
        initial_stage_name: Optional[str] = "Base-Pipeline",
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
            initial_stage_name,
            configs,
            **kwargs
        )
        self.starting_step = 0
        self.log_step = self.num_inference_steps - 1

    def generate(self, latents: torch.FloatTensor) -> List:
        needs_upcasting = (
            self.pipeline.vae.dtype == torch.float16
            and self.pipeline.vae.config.force_upcast
        )
        if needs_upcasting:
            self.pipeline.upcast_vae()
            latents = latents.to(
                next(iter(self.pipeline.vae.post_quant_conv.parameters())).dtype
            )
        images = self.pipeline.vae.decode(
            latents / self.pipeline.vae.config.scaling_factor, return_dict=False
        )[0]
        if needs_upcasting:
            self.pipeline.vae.to(dtype=torch.float16)
        if self.pipeline.watermark is not None:
            images = self.pipeline.watermark.apply_watermark(images)
        images = self.pipeline.image_processor.postprocess(images, output_type="pil")
        self.pipeline.maybe_free_model_hooks()
        return images

    def add_refiner_stage(
        self,
        pipeline: StableDiffusionXLImg2ImgPipeline,
        num_inference_steps: Optional[int] = None,
        strength: Optional[float] = 0.3,
        configs: Optional[Dict] = None,
    ):
        self.strength = strength
        super().add_stage(pipeline, num_inference_steps, "Refiner-Pipeline", configs)
        self.starting_step = 0
        _, self.log_step = self.pipeline.get_timesteps(
            self.num_inference_steps, self.strength, self.pipeline._execution_device
        )
        self.log_step = self.starting_step + self.log_step - 1
