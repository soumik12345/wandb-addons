from typing import Any, Dict, List, Optional, Union

import torch
import wandb
from PIL import Image

from diffusers.image_processor import PipelineImageInput
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
)

from .base import BaseDiffusersBaseCallback, BaseImage2ImageCallback


class StableDiffusionCallback(BaseDiffusersBaseCallback):
    """Callback for [🧨 Diffusers](https://huggingface.co/docs/diffusers/index) logging
    the results of a
    [`StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/v0.9.0/en/api/pipelines/stable_diffusion#diffusers.StableDiffusionPipeline)
    generation to Weights & Biases.

    !!! note "Features:"
        - The callback automatically logs basic configs like prompt, negative prompt,
            etc. along with the generated image in a
            [`wandb.Table`](https://docs.wandb.ai/guides/tables).
        - The callback also logs configs for both the experiment as well as pipelines
            with the wandb run.
        - No need to initialize a run, the callback automatically initialized and ends
            runs gracefully.

    !!! example "Example usage:"
        You can fine an example notebook [here](../examples/stable_diffusion).

        ```python
        import torch
        from diffusers import StableDiffusionPipeline

        from wandb_addons.diffusers import StableDiffusionCallback


        pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        )
        pipeline = pipeline.to("cuda")

        prompt = [
            "a photograph of an astronaut riding a horse",
            "a photograph of a dragon"
        ]
        negative_prompt = ["ugly, deformed", "ugly, deformed"]
        num_images_per_prompt = 2
        configs = {
            "eta": 0.0,
            "guidance_rescale": 0.0,
        }

        # Create the WandB callback for StableDiffusionPipeline
        callback = StableDiffusionCallback(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            wandb_project="diffusers",
            num_images_per_prompt=num_images_per_prompt,
            configs=configs,
        )

        # Add the callback to the pipeline
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            callback=callback,
            num_images_per_prompt=num_images_per_prompt,
            **configs,
        )
        ```

    Arguments:
        pipeline (diffusers.StableDiffusionPipeline): The `StableDiffusionPipeline` from
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
            [weave board](https://docs.wandb.ai/guides/weave) instead of W&B dashboard or
            not. The weave mode logs the configs, generated images and timestamp in a
            [`StreamTable`](https://docs.wandb.ai/guides/weave/streamtable) instead of a
            `wandb.Table` and does not require a wandb run to be initialized in order to
            start logging. This makes it possible to log muliple generations without having
            to initialize or terminate runs. Note that the parameter `wandb_entity` must be
            explicitly specified in order to use weave mode.
        num_inference_steps (int): The number of denoising steps. More denoising steps
            usually lead to a higher quality image at the expense of slower inference.
        num_images_per_prompt (Optional[int]): The number of images to generate per
            prompt.
        guidance_scale (float): Guidance scale as defined in
            [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of
            [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is
            enabled by setting `guidance_scale > 1`. Higher guidance scale encourages
            to generate images that are closely linked to the text `prompt`, usually
            at the expense of lower image quality.
        negative_prompt (Optional[Union[str, List[str]]]): The prompt or prompts not
            to guide the image generation. Ignored when not using guidance
            (i.e., ignored if `guidance_scale` is less than `1`).
        configs (Optional[Dict]): Additional configs for the experiment you want to
            sync, for example, for example, `seed` could be a good config to be passed
            here.
    """

    def __init__(
        self,
        pipeline: Union[DiffusionPipeline, StableDiffusionPipeline],
        prompt: Union[str, List[str]],
        wandb_project: str,
        wandb_entity: Optional[str] = None,
        weave_mode: bool = False,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        configs: Optional[Dict] = None,
        **kwargs,
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
            **kwargs,
        )
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1.0
        additional_configs = {
            "guidance_scale": self.guidance_scale,
            "do_classifier_free_guidance": self.do_classifier_free_guidance,
        }
        if not self.weave_mode:
            wandb.config.update(additional_configs)
        else:
            self.configs.update(additional_configs)

    def build_wandb_table(self) -> None:
        super().build_wandb_table()
        self.table_columns += ["Guidance-Scale", "Do-Classifier-Free-Guidance"]

    def populate_table_row(
        self, prompt: str, negative_prompt: str, image: Image
    ) -> None:
        super().populate_table_row(prompt, negative_prompt, image)
        if not self.weave_mode:
            self.table_row += [self.guidance_scale, self.do_classifier_free_guidance]

    def generate(self, latents: torch.FloatTensor) -> List:
        images = self.pipeline.decode_latents(latents)
        images, _ = self.pipeline.run_safety_checker(
            images, self.pipeline._execution_device, latents.dtype
        )
        images = self.pipeline.numpy_to_pil(images)
        return images


class StableDiffusionImg2ImgCallback(BaseImage2ImageCallback):
    """Callback for [🧨 Diffusers](https://huggingface.co/docs/diffusers/index) logging
    the results of a
    [`StableDiffusionImg2ImgPipeline`](https://huggingface.co/docs/diffusers/v0.21.0/en/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline)
    generation to Weights & Biases.

    !!! note "Features:"
        - The callback automatically logs basic configs like input image, prompt,
            negative prompt, etc. along with the generated image in a
            [`wandb.Table`](https://docs.wandb.ai/guides/tables).
        - The callback also logs configs for both the experiment as well as pipelines
            with the wandb run.
        - No need to initialize a run, the callback automatically initialized and ends
            runs gracefully.

    !!! example "Example usage:"
        You can fine an example notebook [here](../examples/stable_diffusion_img2img).

        ```python
        import requests
        import torch
        from PIL import Image
        from io import BytesIO

        from diffusers import StableDiffusionImg2ImgPipeline

        from wandb_addons.diffusers import StableDiffusionImg2ImgCallback


        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")

        url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((768, 512))

        prompt = "A fantasy landscape, trending on artstation"

        num_images_per_prompt = 1
        strength = 0.75

        callback = StableDiffusionImg2ImgCallback(
            pipeline=pipe, prompt=prompt,
            input_images=init_image,
            wandb_project="diffusers",
            num_images_per_prompt=num_images_per_prompt,
            strength=strength,
        )

        results = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            callback=callback
        )
        ```

    Arguments:
        pipeline (diffusers.StableDiffusionPipeline): The `StableDiffusionPipeline` from
            `diffusers`.
        prompt (Union[str, List[str]]): The prompt or prompts to guide the image
            generation.
        input_images (PipelineImageInput): The input image, numpy array or tensor
            representing an image batch to be used as the starting point. For both numpy
            array and pytorch tensor, the expected value range is between [0, 1] If it's
            a tensor or a list or tensors, the expected shape should be `(B, C, H, W)`
            or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected
            shape should be (B, H, W, C) or (H, W, C) It can also accept image latents as
            image, but if passing latents directly it is not encoded again.
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
            [weave board](https://docs.wandb.ai/guides/weave) instead of W&B dashboard or
            not. The weave mode logs the configs, generated images and timestamp in a
            [`StreamTable`](https://docs.wandb.ai/guides/weave/streamtable) instead of a
            `wandb.Table` and does not require a wandb run to be initialized in order to
            start logging. This makes it possible to log muliple generations without having
            to initialize or terminate runs. Note that the parameter `wandb_entity` must be
            explicitly specified in order to use weave mode.
        num_inference_steps (int): The number of denoising steps. More denoising steps
            usually lead to a higher quality image at the expense of slower inference.
        strength (Optional[float]): Indicates extent to transform the reference image.
            Must be between 0 and 1. image is used as a starting point and more noise
            is added the higher the strength. The number of denoising steps depends on
            the amount of noise initially added. When strength is 1, added noise is
            maximum and the denoising process runs for the full number of iterations
            specified in `num_inference_steps`. A value of 1 essentially ignores image.
        guidance_scale (float): Guidance scale as defined in
            [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of
            [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is
            enabled by setting `guidance_scale > 1`. Higher guidance scale encourages
            to generate images that are closely linked to the text `prompt`, usually
            at the expense of lower image quality.
        num_images_per_prompt (Optional[int]): The number of images to generate per
            prompt.
        negative_prompt (Optional[Union[str, List[str]]]): The prompt or prompts not
            to guide the image generation. Ignored when not using guidance
            (i.e., ignored if `guidance_scale` is less than `1`).
        configs (Optional[Dict]): Additional configs for the experiment you want to
            sync, for example, for example, `seed` could be a good config to be passed
            here.
    """

    def __init__(
        self,
        pipeline: Union[DiffusionPipeline, StableDiffusionImg2ImgPipeline],
        prompt: Union[str, List[str]],
        input_images: PipelineImageInput,
        wandb_project: str,
        wandb_entity: Optional[str] = None,
        weave_mode: bool = False,
        num_inference_steps: int = 50,
        strength: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        configs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pipeline=pipeline,
            prompt=prompt,
            input_images=input_images,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            weave_mode=weave_mode,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            configs=configs,
            **kwargs,
        )
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1.0
        additional_configs = {
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "do_classifier_free_guidance": self.do_classifier_free_guidance,
        }
        if self.weave_mode:
            self.configs.update(additional_configs)
        else:
            wandb.config.update(additional_configs)

    def at_initial_step(self):
        super().at_initial_step()
        _, self.log_step = self.pipeline.get_timesteps(
            self.num_inference_steps, self.strength, self.pipeline._execution_device
        )

    def generate(self, latents: torch.FloatTensor) -> List:
        images = self.pipeline.vae.decode(
            latents / self.pipeline.vae.config.scaling_factor, return_dict=False
        )[0]
        images, has_nsfw_concept = self.pipeline.run_safety_checker(
            images, self.pipeline._execution_device, latents.dtype
        )
        do_denormalize = (
            [True] * images.shape[0]
            if has_nsfw_concept is None
            else [not has_nsfw for has_nsfw in has_nsfw_concept]
        )
        images = self.pipeline.image_processor.postprocess(
            images, output_type="pil", do_denormalize=do_denormalize
        )
        return images
