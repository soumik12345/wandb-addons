from typing import Dict, List, Optional, Union

import torch
from diffusers import (DiffusionPipeline, StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLPipeline)

from ..base import BaseMultiPipelineCallback


class StableDiffusionXLCallback(BaseMultiPipelineCallback):
    """Callback for logging the resulst of a text-to-image workflow with
    [Stable Diffusion XL](https://huggingface.co/docs/diffusers/v0.21.0/en/using-diffusers/sdxl#texttoimage)
    to Weights & Biases.

    !!! note "Features:"
        - The callback automatically logs basic configs like prompt, negative prompt,
            etc. along with the generated image in a
            [`wandb.Table`](https://docs.wandb.ai/guides/tables).
        - The callback also logs configs for both the experiment as well as pipelines
            with the wandb run.
        - No need to initialize a run, the callback automatically initialized and ends
            runs gracefully.
        - Supports logging base and refinement stages of a workflow using a single
            callback.

    !!! example "Example usage:"
        You can fine an example notebook [here](../examples/sdxl).

        ```python
        from functools import partial

        import torch

        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
        from wandb_addons.diffusers import StableDiffusionXLCallback


        base_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

        base_pipeline.enable_model_cpu_offload()

        prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

        num_inference_steps = 50

        callback = StableDiffusionXLCallback(
            pipeline=base_pipeline,
            prompt=prompt,
            wandb_project="diffusers-sdxl",
            wandb_entity="geekyrakshit",
            weave_mode=True,
            num_inference_steps=num_inference_steps,
            initial_stage_name="base",
        )

        image = base_pipeline(
            prompt=prompt,
            output_type="latent",
            num_inference_steps=num_inference_steps,
            callback=partial(callback, end_experiment=False)
        ).images[0]

        refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base_pipeline.text_encoder_2,
            vae=base_pipeline.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner_pipeline.enable_model_cpu_offload()

        num_inference_steps = 50
        strength = 0.3

        callback.add_refiner_stage(
            refiner_pipeline, num_inference_steps=num_inference_steps, strength=strength
        )

        image = refiner_pipeline(
            prompt=prompt, image=image[None, :], callback=callback
        ).images[0]
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
        """Add the refinement stage to the callback to log the results of the refiner
        pipeline.

        Arguments:
            pipeline (diffusers.DiffusionPipeline): The `DiffusionPipeline` from
                for the new stage.
            num_inference_steps (Optional[int]): The number of denoising steps for the
                new stage. More denoising steps usually lead to a higher quality image
                at the expense of slower inference.
            strength (Optional[float]): Conceptually, indicates how much to transform
                the reference image. Must be between 0 and 1. image will be used as a
                starting point, adding more noise to it the larger the strength. The
                number of denoising steps depends on the amount of noise initially
                added. When strength is 1, added noise will be maximum and the
                denoising process will run for the full number of iterations specified
                in num_inference_steps. A value of 1, therefore, essentially ignores
                image. Note that in the case of `denoising_start` being declared as an
                integer, the value of strength will be ignored.
            configs (Optional[Dict]): Additional configs for the new stage you want to
                sync, for example, for example, `seed` could be a good config to be
                passed here.
        """
        self.strength = strength
        super().add_stage(pipeline, num_inference_steps, "Refiner-Pipeline", configs)
        self.starting_step = 0
        _, self.log_step = self.pipeline.get_timesteps(
            self.num_inference_steps, self.strength, self.pipeline._execution_device
        )
        self.log_step = self.starting_step + self.log_step - 1
