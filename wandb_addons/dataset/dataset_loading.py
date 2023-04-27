import os
import shlex
import subprocess
from typing import Any, Dict, Optional, Tuple

import wandb
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.dataset_builder import DatasetBuilder
from tensorflow_datasets.core.dataset_info import DatasetInfo

from ..utils import fetch_wandb_artifact
from .utils import (
    _DATASET_TYPE,
    _build_datasets,
    _change_artifact_dir_name,
    _create_empty_file,
    _get_dataset_name_from_artifact_address,
    _get_dataset_registration_statement,
    _remove_redundant_files,
)


def _load_dataset_from_tfds_module(
    artifact_address: str,
    artifact_dir: str,
    dataset_name: str,
    remove_redundant_data_files: bool = True,
    quiet: bool = False,
) -> Tuple[Dict[str, _DATASET_TYPE], DatasetInfo]:
    wandb.termlog("Creating __init__.py inside `artifact_dir`...")
    _create_empty_file(os.path.join(artifact_dir, "__init__.py"))
    wandb.termlog("Done!")

    # build and prepare the dataset to `~/tensorflow_datasets/
    wandb.termlog(f"Building dataset {dataset_name}...")
    current_working_dir = os.getcwd()
    os.chdir(os.path.join(artifact_dir, dataset_name))
    if quiet:
        tf.get_logger().setLevel("ERROR")
    result = subprocess.call(
        shlex.split("tfds build"),
        stderr=subprocess.DEVNULL if quiet else None,
        stdout=subprocess.DEVNULL if quiet else None,
    )
    os.chdir(current_working_dir)
    if result.returncode != 0:
        raise wandb.Error(f"Unable to load artifact {artifact_address}")
    wandb.termlog(f"Built dataset {dataset_name}!")

    if remove_redundant_data_files:
        wandb.termlog(f"Removing redundant files from {artifact_dir}...")
        _remove_redundant_files(artifact_dir, dataset_name)
        wandb.termlog("Done!")

    try:
        wandb.termlog(f"Registering dataset {dataset_name}...")
        exec(_get_dataset_registration_statement(artifact_dir, dataset_name))
        wandb.termlog(f"Registered dataset {dataset_name}!")
    except ImportError as exception:
        print(exception)
        raise wandb.Error(f"Unable to register {artifact_dir}.{dataset_name}")

    dataset_builder = tfds.builder(dataset_name)
    dataset_builder.download_and_prepare()
    dataset_splits, dataset_builder_info = _build_datasets(dataset_builder)

    return dataset_splits, dataset_builder_info


def load_dataset(
    artifact_address: str,
    artifact_type: Optional[str] = "dataset",
    remove_redundant_data_files: bool = True,
    quiet: bool = False,
) -> Tuple[Dict[str, _DATASET_TYPE], DatasetInfo]:
    """Load a dataset from a [wandb artifact](https://docs.wandb.ai/guides/artifacts).

    Using this function you can load a dataset hosted as a
    [wandb artifact](https://docs.wandb.ai/guides/artifacts) in a single line of code,
    and use our powerful data processing methods to quickly get your dataset ready for
    training in a deep learning model.

    Usage:

    ```python
    from wandb_addons.dataset import load_dataset

    datasets, dataset_builder_info = load_dataset("geekyrakshit/artifact-accessor/monkey_species:v0")
    ```

    !!! example "Example notebooks:"
        - [🔥 Data Loading with WandB Artifacts 🪄🐝](../examples/load_dataset).

    Args:
        artifact_address (str): A human-readable name for the artifact, which is how you can
            identify the artifact in the UI or reference it in
            [`use_artifact`](https://docs.wandb.ai/ref/python/run#use_artifact) calls. Names can
            contain letters, numbers, underscores, hyphens, and dots. The name must be unique across
            a project.
        artifact_type (str): The type of the artifact, which is used to organize and differentiate
            artifacts. Common typesCinclude dataset or model, but you can use any string containing
            letters, numbers, underscores, hyphens, and dots.
        remove_redundant_data_files (bool): Whether to remove the redundant data files from the
            artifacts directory after building the tfrecord dataset.
        quiet (bool): Whether to suppress the output of dataset build process or not.

    Returns:
        (Tuple[Dict[str, tf.data.Dataset], DatasetInfo]): A tuple of dictionary Dictionary mapping
            split aliases to the respective
            [TensorFlow Prefetched dataset](https://www.tensorflow.org/guide/data_performance#prefetching)
            objects and the
            [`tfds.core.DatasetInfo`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo)
            that documents datasets, including its name, version, and features.
    """
    artifact_dir = fetch_wandb_artifact(
        artifact_address=artifact_address, artifact_type=artifact_type
    )
    artifact_dir = _change_artifact_dir_name(artifact_dir)
    dataset_name = _get_dataset_name_from_artifact_address(artifact_address)

    try:
        dataset_builder = tfds.builder_from_directory(artifact_dir)
        dataset_builder.download_and_prepare()
        dataset_splits, dataset_builder_info = _build_datasets(dataset_builder)
    except Exception as e:
        wandb.termwarn(
            "Failed to detect TFRecords in the artifact. Attempting to build tfrecords"
        )
        dataset_splits, dataset_builder_info = _load_dataset_from_tfds_module(
            artifact_address,
            artifact_dir,
            dataset_name,
            remove_redundant_data_files,
            quiet,
        )

    return dataset_splits, dataset_builder_info
