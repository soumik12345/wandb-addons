from typing import Dict, Union, List, Optional

import wandb
from diffusers import StableDiffusionPipeline


class StableDiffusionCallback:
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
            "Step",
            "Prompt",
            "Negative-Prompt",
            "Generated-Image",
            "Guidance-Scale",
            "Do-Classifier-Free-Guidance",
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
            for image in images:
                self.wandb_table.add_data(
                    step,
                    self.prompt,
                    self.negative_prompt,
                    wandb.Image(image),
                    self.guidance_scale,
                    self.do_classifier_free_guidance,
                )
            wandb.log({"Generated-Images": self.wandb_table})
            wandb.finish()
