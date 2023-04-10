# 🐝 Weights & Biases Addons

![](https://github.com/soumik12345/wandb-addons/actions/workflows/tests.yml/badge.svg)

Weights & Biases Addons is a repository that provides of integrations and utilities that will supercharge your [Weights & Biases](https://wandb.ai/site) workflows. Its a repositpry built and maintained by `wandb` users for `wandb` users.

## Integrations

### [🌀 Ciclo](https://github.com/cgarciae/ciclo)

Functional callbacks for experiment tracking on [Weights & Biases](https://wandb.ai/site) with [Ciclo](https://github.com/cgarciae/ciclo).

In order to install `wandb-addons` along with the dependencies for the ciclo callbacks, you can run:

```shell
git clone https://github.com/soumik12345/wandb-addons
pip install .[jax]
```

Once you've installed `wandb-addons`, you can import it using:

```python
from wandb_addons.ciclo import WandbLogger
```

For more information, check out more at the [docs](ciclo/ciclo).

### [MonAI](https://github.com/Project-MONAI/MONAI)

Event handlers for experiment tracking on [Weights & Biases](https://wandb.ai/site) with [MonAI](https://github.com/Project-MONAI/MONAI) Engine for deep learning in healthcare imaging.

In order to install `wandb-addons` along with the dependencies for the ciclo callbacks, you can run:

```shell
git clone https://github.com/soumik12345/wandb-addons
pip install .[monai]
```

Once you've installed `wandb-addons`, you can import it using:

```python
from wandb_addons.monai import WandbStatsHandler, WandbModelCheckpointHandler
```

For more information, check out more at the [docs](monai/monai).

## Status

`wandb-addons` is still in early development, the API for integrations and utilities is subject to change, expect things to break. If you are interested in contributing, please feel free to open an issue and/or raise a pull request.
