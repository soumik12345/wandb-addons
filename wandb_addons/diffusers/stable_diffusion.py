from typing import Dict, List, Optional, Union

import wandb
from diffusers import StableDiffusionPipeline

from .utils import chunkify


class WandBStableDiffusionCallback:
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


        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")

        prompt = [
            "a photograph of an astronaut riding a horse",
            "a photograph of a dragon",
        ]

        # Create the WandB callback for StableDiffusionPipeline
        callback = WandBStableDiffusionCallback(
            pipe, prompt=prompt, wandb_project="diffusers", num_images_per_prompt=2
        )

        # Add the callback to the pipeline
        results = pipe(prompt, callback=callback, num_images_per_prompt=2)
        ```

    Arguments:
        pipe (diffusers.StableDiffusionPipeline): The `StableDiffusionPipeline` from
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
            sync, for example, seed could be a good config to be passed here.
    """

    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        prompt: Union[str, List[str]],
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        configs: Optional[Dict] = None,
    ):
        self.pipe = pipe
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.num_images_per_prompt = num_images_per_prompt
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1.0
        self.negative_prompt = negative_prompt
        self.configs = configs
        self.initialize_wandb(wandb_project, wandb_entity)
        self.build_wandb_table()

    def initialize_wandb(self, wandb_project, wandb_entity):
        if wandb.run is None:
            if wandb_project is not None:
                additional_configs = {
                    "prompt": self.prompt,
                    "negative_prompt": self.negative_prompt,
                    "guidance_scale": self.guidance_scale,
                    "do_classifier_free_guidance": self.do_classifier_free_guidance,
                    "pipe": dict(self.pipe.config),
                }
                self.configs = (
                    {**self.configs, **additional_configs}
                    if self.configs is not None
                    else additional_configs
                )
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    job_type="text-to-image",
                    config=self.configs,
                )
            else:
                wandb.termerror("The parameter wandb_project must be provided.")

    def build_wandb_table(self):
        self.table_columns = [
            "Prompt",
            "Negative-Prompt",
            "Generated-Image",
            "Guidance-Scale",
            "Do-Classifier-Free-Guidance",
            "Inference-Step",
        ]
        self.wandb_table = wandb.Table(columns=self.table_columns)

    def generate(self, latents):
        text_embeddings = self.pipe._encode_prompt(
            self.prompt,
            self.pipe._execution_device,
            self.num_images_per_prompt,
            self.do_classifier_free_guidance,
            self.negative_prompt,
        )
        images = self.pipe.decode_latents(latents)
        images, _ = self.pipe.run_safety_checker(
            images, self.pipe._execution_device, text_embeddings.dtype
        )
        images = self.pipe.numpy_to_pil(images)
        return images

    def __call__(self, step, timestep, latents):
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
                    self.wandb_table.add_data(
                        prompt_logging[idx],
                        negative_prompt_logging[idx],
                        wandb.Image(image),
                        self.guidance_scale,
                        self.do_classifier_free_guidance,
                        step,
                    )
            wandb.log({"Generated-Images": self.wandb_table})
            wandb.finish()
