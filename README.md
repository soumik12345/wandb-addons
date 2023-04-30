# 🐝 Weights & Biases Addons

![Tests](https://github.com/soumik12345/wandb-addons/actions/workflows/tests.yml/badge.svg)
![Deploy](https://github.com/soumik12345/wandb-addons/actions/workflows/deploy.yml/badge.svg)

Weights & Biases Addons is a repository that provides of integrations and utilities that will supercharge your [Weights & Biases](https://wandb.ai/site) workflows. Its a repositpry built and maintained by `wandb` users for `wandb` users.

## WandB Datasets

A set of utilities for easily accessing datasets for various machine learning tasks using [Weights & Biases artifacts](https://docs.wandb.ai/guides/artifacts).

- **[`WandbDatasetBuilder`](https://soumik12345.github.io/wandb-addons/dataset/dataset_loading/#wandb_addons.dataset.dataset_builder.WandbDatasetBuilder):** An abstract class for Dataset builder that enables building a dataset and upload it as a [Weights & Biases Artifact](https://docs.wandb.ai/guides/artifacts).

- **[`upload_dataset`](https://soumik12345.github.io/wandb-addons/dataset/dataset_loading/#wandb_addons.dataset.dataset_upload.upload_dataset):** Upload and register a dataset with a TFDS module or a TFDS builder script as a Weights & Biases artifact. This function would verify if a TFDS build/registration is possible with the current specified dataset path and upload it as a Weights & Biases artifact.

- **[`load_dataset`](https://soumik12345.github.io/wandb-addons/dataset/dataset_loading/#wandb_addons.dataset.dataset_loading.load_dataset):** Load a dataset from a wandb artifact. Using this function you can load a dataset hosted as a wandb artifact in a single line of code, and use our powerful data processing methods to quickly get your dataset ready for training in a deep learning model.

## Integrations

### [🌀 Ciclo](https://github.com/cgarciae/ciclo)

Functional callbacks for experiment tracking on [Weights & Biases](https://wandb.ai/site) with [Ciclo](https://github.com/cgarciae/ciclo).

In order to install `wandb-addons` along with the dependencies for the ciclo callbacks, you can run:

```shell
git clone https://github.com/soumik12345/wandb-addons
pip install wandb-addons[jax]
```

Once you've installed `wandb-addons`, you can import it using:

```python
from wandb_addons.ciclo import WandbLogger
```

For more information, check out more at the [docs](https://soumik12345.github.io/wandb-addons/ciclo/ciclo/).

### [MonAI](https://github.com/Project-MONAI/MONAI)

Event handlers for experiment tracking on [Weights & Biases](https://wandb.ai/site) with [MonAI](https://github.com/Project-MONAI/MONAI) Engine for deep learning in healthcare imaging.

In order to install `wandb-addons` along with the dependencies for the ciclo callbacks, you can run:

```shell
git clone https://github.com/soumik12345/wandb-addons
pip install wandb-addons[monai]
```

Once you've installed `wandb-addons`, you can import it using:

```python
from wandb_addons.monai import WandbStatsHandler, WandbModelCheckpointHandler
```

For more information, check out more at the [docs](https://soumik12345.github.io/wandb-addons/monai/monai/).

## Status

`wandb-addons` is still in early development, the API for integrations and utilities is subject to change, expect things to break. If you are interested in contributing, please feel free to open an issue and/or raise a pull request.
