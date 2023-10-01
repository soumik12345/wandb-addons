from typing import Dict, List, Optional, Union

import wandb
from diffusers import DiffusionPipeline

from .kandinsky import KandinskyCallback
from .stable_diffusion import StableDiffusionCallback


def get_wandb_callback(
    pipeline: DiffusionPipeline,
    prompt: Union[str, List[str]],
    wandb_project: str,
    num_inference_steps: int,
    num_images_per_prompt: int,
    wandb_entity: Optional[str] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    configs: Dict = {},
    **kwargs,
):
    """A function for automatically inferring the W&B callback for the respective
    `DiffusionPipeline`.

    !!! warning
        While using this function to automatically infer the type of the
        `DiffusionPipeline`, we must ensure to explicitly set the parameters of
        the respective callback exclusive to that particular callback, for example,
        in order to use the `StableDiffusionCallback` we must explicitly pass the
        `guidance_scale`.

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
        num_inference_steps (int): The number of denoising steps. More denoising steps
            usually lead to a higher quality image at the expense of slower inference.
        num_images_per_prompt (Optional[int]): The number of images to generate per
            prompt.
        negative_prompt (Optional[Union[str, List[str]]]): The prompt or prompts not
            to guide the image generation. Ignored when not using guidance
            (i.e., ignored if `guidance_scale` is less than `1`).
        configs (Optional[Dict]): Additional configs for the experiment you want to
            sync, for example, seed could be a good config to be passed here.
    """
    pipeline_name = pipeline.__class__.__name__
    if pipeline_name == "StableDiffusionPipeline":
        return StableDiffusionCallback(
            pipeline=pipeline,
            prompt=prompt,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            configs=configs,
            **kwargs,
        )
    elif pipeline_name in ["KandinskyCombinedPipeline", "KandinskyPipeline"]:
        return KandinskyCallback(
            pipeline=pipeline,
            prompt=prompt,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            configs=configs,
            **kwargs,
        )
    else:
        wandb.Error(f"{pipeline_name} is not supported currently.")
