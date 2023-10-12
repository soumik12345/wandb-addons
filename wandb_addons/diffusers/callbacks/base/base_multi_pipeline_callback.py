from typing import Any, Dict, List, Optional, Union

import wandb
from diffusers import DiffusionPipeline

from .base_callback import BaseDiffusersCallback


class BaseMultiPipelineCallback(BaseDiffusersCallback):
    """Base callback for [🧨 Diffusers](https://huggingface.co/docs/diffusers/index)
    logging the results of multiple
    [`DiffusionPipeline`](https://github.com/huggingface/diffusers/blob/v0.21.0/src/diffusers/pipelines/pipeline_utils.py#L480)s
    to Weights & Biases.

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
        num_inference_steps (int): The number of denoising steps. More denoising steps
            usually lead to a higher quality image at the expense of slower inference.
        num_images_per_prompt (Optional[int]): The number of images to generate per
            prompt.
        negative_prompt (Optional[Union[str, List[str]]]): The prompt or prompts not
            to guide the image generation. Ignored when not using guidance
            (i.e., ignored if `guidance_scale` is less than `1`).
        initial_stage_name (Optional[str]): The name of the initial stage. If not specified,
            it would be set to `"stage_1"`.
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
        weave_mode: bool = False,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        initial_stage_name: Optional[str] = None,
        configs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        self.stage_name = (
            initial_stage_name if initial_stage_name is not None else "stage_1"
        )
        self.stage_counter = 1
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

    def update_configs(self) -> None:
        """Update the configs as a state of the callback. This function is called inside
        `initialize_wandb()`. In this function, the configs regarding the base pipeline
        are updated as well.
        """
        additional_configs = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_images_per_prompt": self.num_images_per_prompt,
            self.stage_name: {
                "pipeline": dict(self.pipeline.config),
                "num_inference_steps": self.num_inference_steps,
            },
        }
        self.configs = (
            {**self.configs, **additional_configs}
            if self.configs is not None
            else additional_configs
        )

    def add_stage(
        self,
        pipeline: DiffusionPipeline,
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
        self.pipeline = pipeline
        self.num_inference_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else self.num_inference_steps
        )
        self.stage_counter += 1
        self.stage_name = (
            stage_name if stage_name is not None else f"stage_{self.stage_counter}"
        )
        additional_configs = {
            self.stage_name: {
                "pipeline": dict(self.pipeline.config),
                "num_inference_steps": self.num_inference_steps,
                "stage-sequence": self.stage_counter,
            }
        }
        if configs is not None:
            additional_configs[self.stage_name].update(configs)
        self.configs.update(additional_configs)
        if wandb.run is not None:
            wandb.config.update(additional_configs)

    def at_initial_step(self):
        """A function that will be called at the initial step of the denoising loop during inference."""
        if self.stage_counter == 1:
            super().at_initial_step()

    def build_wandb_table(self) -> None:
        """Specifies the columns of the wandb table if not in weave mode. This function is
        called automatically when the callback is initialized.
        """
        super().build_wandb_table()
        self.table_columns = ["Stage-Sequence", "Stage-Name"] + self.table_columns

    def populate_table_row(self, prompt: str, negative_prompt: str, image: Any) -> None:
        """Populates the table row with the prompt, negative prompt, the generated image, and
        the configs.

        Arguments:
            prompt (str): The prompt to guide the image generation.
            negative_prompt (str): The prompt not to guide the image generation.
            image (Image): The generated image.
        """
        super().populate_table_row(prompt, negative_prompt, image)
        if self.weave_mode:
            self.table_row.update(
                {"Stage-Sequence": self.stage_counter, "Stage-Name": self.stage_name}
            )
        else:
            self.table_row = [self.stage_counter, self.stage_name] + self.table_row
