# Callbacks for 🧨 Diffusers pipelines

Callbacks for logging experiment details, configs and generated images for multi-modal diffusion pipelines from [Diffusers 🧨](https://huggingface.co/docs/diffusers) to your [Weights & Biases workspace](https://docs.wandb.ai/guides/app/pages/workspaces) or [Weave Dashboard](https://weave.wandb.ai/).

In order to install the depensencies to use the integration, you can run:

```shell
git clone https://github.com/soumik12345/wandb-addons
pip install ./wandb-addons[huggingface]
```

For detailed documentation, check out the following:

- [Auto-integrate](./auto_integrate.md)
- [DeepFloyd IF Callbacks](./deepfloyd_if.md)
- [Kandinsky Callback](./kandinsky.md)
- [Stable Diffusion Callbacks](./stable_diffusion.md)

|Callback|Run-in-Colab|WandB Run|
|---|---|---|
|Stable Diffusion|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/wandb-addons/blob/main/docs/diffusers/examples/stable_diffusion.ipynb)|[![](../assets/wandb-github-badge-gradient.svg)](https://wandb.ai/geekyrakshit/diffusers-new/runs/gii2kqqr)|
|Stable Diffusion XL|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/wandb-addons/blob/main/docs/diffusers/examples/sdxl.ipynb)|[![](../assets/wandb-github-badge-gradient.svg)](https://wandb.ai/geekyrakshit/diffusers-new/runs/ygx2dldj)|
|Stable Diffusion Image2Image|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/wandb-addons/blob/main/docs/diffusers/examples/stable_diffusion_img2img.ipynb)|[![](../assets/wandb-github-badge-gradient.svg)](https://wandb.ai/geekyrakshit/diffusers-2/runs/a4a7148w)|
|Kandinsky v2.1|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/wandb-addons/blob/main/docs/diffusers/examples/kandinsky.ipynb)|[![](../assets/wandb-github-badge-gradient.svg)](https://wandb.ai/geekyrakshit/diffusers/runs/1dle2dz8)|
|DeepFloyd IF|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/wandb-addons/blob/main/docs/diffusers/examples/deepfloyd_if.ipynb)|[![](../assets/wandb-github-badge-gradient.svg)](https://wandb.ai/geekyrakshit/diffusers-2/runs/ac4or5q6)|


| Text-to-Image on Weights & Biases |
| -------- |
| ![](./assets/text_to_image.png) |

| Image-to-Image on Weights & Biases |
| -------- |
| ![](./assets/image_to_image.png) |

| Multi-pipeline Text-to-Image Experiments using DeepFloydIF on Weights & Biases |
| -------- |
| ![](./assets/multi_pipeline_t2i.png) |
