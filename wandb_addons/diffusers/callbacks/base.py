from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import wandb
import numpy as np
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.image_processor import PipelineImageInput

from .utils import chunkify


class BaseDiffusersBaseCallback(ABC):
    """Base callback for [🧨 Diffusers](https://huggingface.co/docs/diffusers/index)
    logging the results of a
    [`DiffusionPipeline`](https://github.com/huggingface/diffusers/blob/v0.21.0/src/diffusers/pipelines/pipeline_utils.py#L480)
    generation to Weights & Biases.

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
            sync, for example, for example, `seed` could be a good config to be passed
            here.
    """

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        prompt: Union[str, List[str]],
        wandb_project: str,
        wandb_entity: Optional[str] = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        configs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.prompt = prompt
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.num_inference_steps = num_inference_steps
        self.num_images_per_prompt = num_images_per_prompt
        self.negative_prompt = negative_prompt
        self.configs = configs
        self.wandb_table = None
        self.table_row = []
        self.starting_step = 1
        self.log_step = num_inference_steps
        self.job_type = "text-to-image"
        self.initialize_wandb(wandb_project, wandb_entity)
        self.build_wandb_table()

    def update_configs(self) -> None:
        additional_configs = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "num_images_per_prompt": self.num_images_per_prompt,
            "pipeline": dict(self.pipeline.config),
        }
        self.configs = (
            {**self.configs, **additional_configs}
            if self.configs is not None
            else additional_configs
        )

    def initialize_wandb(self, wandb_project, wandb_entity) -> None:
        if wandb.run is None:
            if wandb_project is not None:
                self.update_configs()
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    job_type=self.job_type,
                    config=self.configs,
                )
            else:
                wandb.termerror("The parameter wandb_project must be provided.")

    def build_wandb_table(self) -> None:
        self.table_columns = ["Prompt", "Negative-Prompt", "Generated-Image"]

    @abstractmethod
    def generate(self, latents: torch.FloatTensor) -> List:
        pass

    def populate_table_row(
        self, prompt: str, negative_prompt: str, image: Image
    ) -> None:
        self.table_row = [prompt, negative_prompt, wandb.Image(image)]

    def at_initial_step(self):
        self.wandb_table = wandb.Table(columns=self.table_columns)

    def __call__(self, step: int, timestep: int, latents: torch.FloatTensor):
        if step == self.starting_step:
            self.at_initial_step()
        if step == self.log_step:
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
                    self.populate_table_row(
                        prompt_logging[idx], negative_prompt_logging[idx], image
                    )
                    self.wandb_table.add_data(*self.table_row)
            wandb.log({"Text-To-Image": self.wandb_table})
            wandb.finish()


class BaseImage2ImageCallback(BaseDiffusersBaseCallback):
    """Base callback for [🧨 Diffusers](https://huggingface.co/docs/diffusers/index)
    logging the results of a
    [`DiffusionPipeline`](https://github.com/huggingface/diffusers/blob/v0.21.0/src/diffusers/pipelines/pipeline_utils.py#L480)
    for image2image translation task to Weights & Biases.

    Arguments:
        pipeline (diffusers.DiffusionPipeline): The `DiffusionPipeline` from
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
        num_inference_steps (int): The number of denoising steps. More denoising steps
            usually lead to a higher quality image at the expense of slower inference.
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
        pipeline: DiffusionPipeline,
        prompt: Union[str, List[str]],
        input_images: PipelineImageInput,
        wandb_project: str,
        wandb_entity: Optional[str] = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        configs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        self.job_type = "image-to-image"
        super().__init__(
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
        self.input_images = self.postprocess_input_images(input_images)

    def postprocess_input_images(self, input_images):
        if isinstance(input_images, torch.Tensor):
            input_images = self.pipeline.image_processor.pt_to_numpy(input_images)
            input_images = self.pipeline.image_processor.numpy_to_pil(input_images)
        elif isinstance(input_images, Image.Image):
            input_images = [input_images]
        elif isinstance(input_images, np.array):
            input_images = self.pipeline.image_processor.numpy_to_pil(input_images)
        return input_images

    def build_wandb_table(self) -> None:
        super().build_wandb_table()
        self.table_columns = ["Input-Image"] + self.table_columns

    def populate_table_row(
        self, input_image: Image.Image, prompt: str, negative_prompt: str, image: Any
    ) -> None:
        self.table_row = [
            wandb.Image(input_image),
            prompt,
            negative_prompt,
            wandb.Image(image),
        ]

    def __call__(self, step: int, timestep: int, latents: torch.FloatTensor):
        if step == self.starting_step:
            self.at_initial_step()
        if step == self.log_step:
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
                    self.populate_table_row(
                        self.input_images[0],
                        prompt_logging[idx],
                        negative_prompt_logging[idx],
                        image,
                    )
                    self.wandb_table.add_data(*self.table_row)
            wandb.log({"Image-To-Image": self.wandb_table})
            wandb.finish()
