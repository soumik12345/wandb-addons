from typing import Dict, List, Optional, Union

import torch
from diffusers import (
    IFPipeline,
    IFSuperResolutionPipeline,
    StableDiffusionUpscalePipeline,
)

from .base import BaseMultiPipelineCallback


class IFCallback(BaseMultiPipelineCallback):
    """Callback for logging the resulst of a text-to-image workflow with
    [DeepFloyd IF](https://huggingface.co/docs/diffusers/v0.21.0/en/api/pipelines/deepfloyd_if#texttoimage-generation)
    to Weights & Biases.

    !!! note "Features:"
        - The callback automatically logs basic configs like prompt, negative prompt,
            etc. along with the generated image in a
            [`wandb.Table`](https://docs.wandb.ai/guides/tables).
        - The callback also logs configs for both the experiment as well as pipelines
            with the wandb run.
        - No need to initialize a run, the callback automatically initialized and ends
            runs gracefully.
        - Supports logging multiple stages of a workflow using a single callback.

    !!! example "Example usage:"
        You can fine an example notebook [here](../examples/deepfloyd_if).

        ```python
        import gc
        from functools import partial

        import torch

        from diffusers import IFPipeline, IFSuperResolutionPipeline, StableDiffusionUpscalePipeline
        from wandb_addons.diffusers import IFCallback

        # Stage 1
        pipeline_1 = IFPipeline.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
        )
        pipeline_1.enable_model_cpu_offload()

        prompt = 'a photo of a smiling bee wearing a yellow hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "Weights and Biases"'
        prompt_embeds, negative_embeds = pipeline_1.encode_prompt(prompt)
        num_images_per_prompt = 2
        num_inference_steps = 100
        configs = {"guidance_scale": 7.0}

        callback = IFCallback(
            pipeline=pipeline_1,
            prompt=prompt,
            wandb_project="diffusers-2",
            wandb_entity="geekyrakshit",
            weave_mode=False,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            configs=configs
        )

        image = pipeline_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            output_type="pt",
            # Do not end the experiment after the first stage
            callback=partial(callback, end_experiment=False),
            **configs,
        ).images

        # Stage 2
        pipeline_2 = IFSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16
        )
        pipeline_2.enable_model_cpu_offload()

        num_inference_steps = 50

        callback.add_stage(pipeline_2, num_inference_steps=num_inference_steps)

        image = pipeline_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=num_inference_steps,
            output_type="pt",
            callback=partial(callback, end_experiment=False),
        ).images

        # Upscale stage
        safety_modules = {
            "feature_extractor": pipeline_1.feature_extractor,
            "safety_checker": pipeline_1.safety_checker,
            "watermarker": pipeline_1.watermarker,
        }
        pipeline_3 = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            **safety_modules,
            torch_dtype=torch.float16
        )
        pipeline_3.enable_model_cpu_offload()

        num_inference_steps = 75

        callback.add_stage(
            pipeline_3, num_inference_steps=num_inference_steps, stage_name="Upscale"
        )

        image = pipeline_3(
            prompt=prompt,
            image=image,
            noise_level=100,
            num_inference_steps=num_inference_steps,
            callback=callback,
        ).images
        ```

    Arguments:
        pipeline (diffusers.DiffusionPipeline): The `DiffusionPipeline` from
            `diffusers`.
        prompt (Union[str, List[str]]): The prompt or prompts to guide the image
            generation.
        wandb_project (Optional[str]): The name of the project where you're sending
            the new run. The project is not necessary to be specified unless the run
            has automatically been initiatlized before the callback is defined.
        wandb_entity (Optional[str]): An entity is a username or team name where
            you're sending runs. This entity must exist before you can send runs there,
            so make sure to create your account or team in the UI before starting to
            log runs. If you don't specify an entity, the run will be sent to your
            default entity, which is usually your username. Change your default entity
            in [your settings](https://wandb.ai/settings) under "default location to
            create new projects".
        weave_mode (bool): Whether to use log to a
            [weave board](https://docs.wandb.ai/guides/weave) instead of W&B dashboard
            or not. The weave mode logs the configs, generated images and timestamp in a
            [`StreamTable`](https://docs.wandb.ai/guides/weave/streamtable) instead of a
            `wandb.Table` and does not require a wandb run to be initialized in order to
            start logging. This makes it possible to log muliple generations without
            having to initialize or terminate runs. Note that the parameter
            `wandb_entity` must be explicitly specified in order to use weave mode.
        num_inference_steps (int): The number of denoising steps. More denoising steps
            usually lead to a higher quality image at the expense of slower inference.
        num_images_per_prompt (Optional[int]): The number of images to generate per
            prompt.
        negative_prompt (Optional[Union[str, List[str]]]): The prompt or prompts not
            to guide the image generation. Ignored when not using guidance
            (i.e., ignored if `guidance_scale` is less than `1`).
        initial_stage_name (Optional[str]): The name of the initial stage. If not
            specified, it would be set to `"stage_1"`.
        configs (Optional[Dict]): Additional configs for the experiment you want to
            sync, for example, for example, `seed` could be a good config to be passed
            here.
    """

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
        initial_stage_name: Optional[str] = None,
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
            initial_stage_name,
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
        """Add a new stage to the callback to log the results of a new pipeline in a
        multi-pipeline workflow.

        Arguments:
            pipeline (diffusers.DiffusionPipeline): The `DiffusionPipeline` from
                for the new stage.
            num_inference_steps (Optional[int]): The number of denoising steps for the
                new stage. More denoising steps usually lead to a higher quality image
                at the expense of slower inference.
            stage_name (Optional[str]): The name of the new stage. If not specified,
                it would be set to `"stage_{stage_counter}"`.
            configs (Optional[Dict]): Additional configs for the new stage you want to
                sync, for example, for example, `seed` could be a good config to be
                passed here.
        """
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
