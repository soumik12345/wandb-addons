from typing import Dict, List, Optional, Union

import torch
from diffusers import (
    IFPipeline,
    IFSuperResolutionPipeline,
    StableDiffusionUpscalePipeline,
)

from .base import BaseMultiPipelineCallback


class IFCallback(BaseMultiPipelineCallback):
    def __init__(
        self,
        pipeline: Union[
            IFPipeline, IFSuperResolutionPipeline, StableDiffusionUpscalePipeline
        ],
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
        if isinstance(self.pipeline, IFPipeline) or isinstance(
            self.pipeline, IFSuperResolutionPipeline
        ):
            images = (latents / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
            (
                images,
                nsfw_detected,
                watermark_detected,
            ) = self.pipeline.run_safety_checker(
                images, self.pipeline._execution_device, latents.dtype
            )
            images = self.pipeline.numpy_to_pil(images)
            return images
        elif isinstance(self.pipeline, StableDiffusionUpscalePipeline):
            needs_upcasting = (
                self.pipeline.vae.dtype == torch.float16
                and self.pipeline.vae.config.force_upcast
            )
            if needs_upcasting:
                self.pipeline.upcast_vae()
                latents = latents.to(
                    next(iter(self.pipeline.vae.post_quant_conv.parameters())).dtype
                )
            image = self.pipeline.vae.decode(
                latents / self.pipeline.vae.config.scaling_factor, return_dict=False
            )[0]
            if needs_upcasting:
                self.pipeline.vae.to(dtype=torch.float16)
            image, nsfw_detected, watermark_detected = self.pipeline.run_safety_checker(
                image, self.pipeline._execution_device, latents.dtype
            )
            do_denormalize = (
                [True] * image.shape[0]
                if nsfw_detected is None
                else [not has_nsfw for has_nsfw in nsfw_detected]
            )
            image = self.pipeline.image_processor.postprocess(
                image, output_type="pil", do_denormalize=do_denormalize
            )
            image = (
                self.pipeline.watermarker.apply_watermark(image)
                if self.pipeline.watermarker is not None
                else image
            )
            if (
                hasattr(self, "final_offload_hook")
                and self.pipeline.final_offload_hook is not None
            ):
                self.pipeline.final_offload_hook.offload()
            return image

    def add_stage(
        self,
        pipeline: Union[IFPipeline, IFSuperResolutionPipeline],
        num_inference_steps: Optional[int] = None,
        stage_name: Optional[str] = None,
        configs: Optional[Dict] = None,
    ) -> None:
        assert (
            isinstance(pipeline, IFPipeline)
            or isinstance(pipeline, IFSuperResolutionPipeline)
            or isinstance(pipeline, StableDiffusionUpscalePipeline)
        ), "IFCallback only supports IFPipeline and IFSuperResolutionPipeline"
        super().add_stage(pipeline, num_inference_steps, stage_name, configs)
        if isinstance(pipeline, IFSuperResolutionPipeline) or isinstance(
            pipeline, StableDiffusionUpscalePipeline
        ):
            self.starting_step = 0
            self.log_step = self.starting_step + self.num_inference_steps - 1
