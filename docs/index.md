# 🐝 Weights & Biases Addons

![Tests](https://github.com/soumik12345/wandb-addons/actions/workflows/tests.yml/badge.svg)
![Deploy](https://github.com/soumik12345/wandb-addons/actions/workflows/deploy.yml/badge.svg)

Weights & Biases Addons is a repository that provides of integrations and utilities that will supercharge your [Weights & Biases](https://wandb.ai/site) workflows. Its a repositpry built and maintained by `wandb` users for `wandb` users.

## Integrations

### [TensorFlow Datasets](https://www.tensorflow.org/datasets)

A set of utilities for easily accessing datasets for various machine learning tasks using [Weights & Biases artifacts](https://docs.wandb.ai/guides/artifacts) built on top of [**TensorFlow Datasets**](https://www.tensorflow.org/datasets).

In order to install `wandb-addons` along with the dependencies for the dataset utilities, you can run:

```shell
git clone https://github.com/soumik12345/wandb-addons
pip install ./wandb-addons[dataset]
```

- **[`WandbDatasetBuilder`](https://soumik12345.github.io/wandb-addons/dataset/dataset_loading/#wandb_addons.dataset.dataset_builder.WandbDatasetBuilder):** An abstract class for Dataset builder that enables building a dataset and upload it as a [Weights & Biases Artifact](https://docs.wandb.ai/guides/artifacts).

- **[`upload_dataset`](https://soumik12345.github.io/wandb-addons/dataset/dataset_loading/#wandb_addons.dataset.dataset_upload.upload_dataset):** Upload and register a dataset with a TFDS module or a TFDS builder script as a Weights & Biases artifact. This function would verify if a TFDS build/registration is possible with the current specified dataset path and upload it as a Weights & Biases artifact.

- **[`load_dataset`](https://soumik12345.github.io/wandb-addons/dataset/dataset_loading/#wandb_addons.dataset.dataset_loading.load_dataset):** Load a dataset from a wandb artifact. Using this function you can load a dataset hosted as a wandb artifact in a single line of code, and use our powerful data processing methods to quickly get your dataset ready for training in a deep learning model.

### [🌀 Ciclo](https://github.com/cgarciae/ciclo)

Functional callbacks for experiment tracking on [Weights & Biases](https://wandb.ai/site) with [Ciclo](https://github.com/cgarciae/ciclo).

In order to install `wandb-addons` along with the dependencies for the ciclo callbacks, you can run:

```shell
git clone https://github.com/soumik12345/wandb-addons
pip install ./wandb-addons[jax]
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
pip install ./wandb-addons[monai]
```

Once you've installed `wandb-addons`, you can import it using:

```python
from wandb_addons.monai import WandbStatsHandler, WandbModelCheckpointHandler
```

For more information, check out more at the [docs](monai/monai).

## Converting IPython Notebooks to [Reports](https://docs.wandb.ai/guides/reports)

A set of utilities to convert an IPython notebook to a Weights & Biases report.

Simply install `wandb-addons` using

```shell
git clone https://github.com/soumik12345/wandb-addons
pip install ./wandb-addons
```

You can convert your notebook to a report using either the Python function or the CLI:

=== "CLI"
    ```shell
    nb2report \\
        --filepath Use_WandbMetricLogger_in_your_Keras_workflow.ipynb \\
        --wandb_project report-to-notebook \\
        --wandb_entity geekyrakshit \\
        --report_title "Use WandbMetricLogger in your Keras Workflow" \\
        --description "A guide to using the WandbMetricLogger callback in your Keras and TensorFlow training worflow" \\
        --width "readable"
    ```

=== "Python API"
    ```python
    from wandb_addons.report import convert_to_wandb_report

    convert_to_wandb_report(
        filepath="Use_WandbMetricLogger_in_your_Keras_workflow.ipynb",
        wandb_project="report-to-notebook",
        wandb_entity="geekyrakshit",
        report_title="Use WandbMetricLogger in your Keras Workflow",
        description="A guide to using the WandbMetricLogger callback in your Keras and TensorFlow training worflow",
        width="readable"
    )
    ```

For more information, check out more at the [docs](report).

## Status

`wandb-addons` is still in early development, the API for integrations and utilities is subject to change, expect things to break. If you are interested in contributing, please feel free to open an issue and/or raise a pull request.
