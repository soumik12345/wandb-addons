from typing import Dict, List, Optional, Union

import wandb
from diffusers import DiffusionPipeline

from .callbacks import (
    IFCallback,
    KandinskyCallback,
    StableDiffusionCallback,
    StableDiffusionImg2ImgCallback,
    StableDiffusionXLCallback,
)


def get_wandb_callback(
    pipeline: DiffusionPipeline,
    prompt: Union[str, List[str]],
    wandb_project: str,
    num_inference_steps: Optional[int] = None,
    num_images_per_prompt: Optional[int] = None,
    wandb_entity: Optional[str] = None,
    weave_mode: bool = False,
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
        weave_mode (bool): Whether to use log to a
            [weave board](https://docs.wandb.ai/guides/weave) instead of W&B dashboard or
            not. The weave mode logs the configs, generated images and timestamp in a
            [`StreamTable`](https://docs.wandb.ai/guides/weave/streamtable) instead of a
            `wandb.Table` and does not require a wandb run to be initialized in order to
            start logging. This makes it possible to log muliple generations without having
            to initialize or terminate runs. Note that the parameter `wandb_entity` must be
            explicitly specified in order to use weave mode.
        num_inference_steps (Optional[int]): The number of denoising steps. More denoising steps
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
    kwargs = {
        "pipeline": pipeline,
        "prompt": prompt,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "weave_mode": weave_mode,
        "negative_prompt": negative_prompt,
        "configs": configs,
        **kwargs,
    }
    if num_inference_steps is not None:
        kwargs["num_inference_steps"] = num_inference_steps
    if num_images_per_prompt is not None:
        kwargs["num_images_per_prompt"] = num_images_per_prompt
    if pipeline_name == "StableDiffusionPipeline":
        return StableDiffusionCallback(**kwargs)
    elif pipeline_name == "StableDiffusionImg2ImgPipeline":
        return StableDiffusionImg2ImgCallback(**kwargs)
    elif pipeline_name in ["KandinskyCombinedPipeline", "KandinskyPipeline"]:
        return KandinskyCallback(**kwargs)
    elif pipeline_name == "IFPipeline":
        return IFCallback(**kwargs)
    elif pipeline_name == "StableDiffusionXLPipeline":
        return StableDiffusionXLCallback(**kwargs)
    else:
        wandb.Error(f"{pipeline_name} is not supported currently.")
