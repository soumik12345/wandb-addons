import os
import shlex
import subprocess
from typing import Dict, Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.dataset_info import DatasetInfo
from tensorflow_datasets.core.dataset_builder import DatasetBuilder

import wandb

from ..utils import fetch_wandb_artifact
from .utils import (
    _change_artifact_dir_name,
    _create_empty_file,
    _get_dataset_name_from_artifact_address,
    _get_dataset_registration_statement,
    _remove_redundant_files,
    _build_datasets,
)


def load_dataset(
    artifact_address: str,
    artifact_type: Optional[str] = "dataset",
    remove_redundant_data_files: bool = True,
    quiet: bool = False,
) -> Tuple[Dict[str, tf.data.Dataset], DatasetInfo]:
    """Load a dataset from a [wandb artifact](https://docs.wandb.ai/guides/artifacts).

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
    """
    artifact_dir = fetch_wandb_artifact(
        artifact_address=artifact_address, artifact_type=artifact_type
    )
    artifact_dir = _change_artifact_dir_name(artifact_dir)

    wandb.termlog("Creating __init__.py inside `artifact_dir`...")
    _create_empty_file(os.path.join(artifact_dir, "__init__.py"))
    wandb.termlog("Done!")

    dataset_name = _get_dataset_name_from_artifact_address(artifact_address)

    # build and prepare the dataset to `~/tensorflow_datasets/
    wandb.termlog(f"Building dataset {dataset_name}...")
    current_working_dir = os.getcwd()
    os.chdir(os.path.join(artifact_dir, dataset_name))
    if not quiet:
        subprocess.call(shlex.split("tfds build"))
    else:
        tf.get_logger().setLevel("ERROR")
        subprocess.call(
            shlex.split("tfds build"),
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
    os.chdir(current_working_dir)
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

    dataset_builder = tfds.builder("monkey_species")
    dataset_builder.download_and_prepare()
    dataset_splits, dataset_builder_info = _build_datasets(dataset_builder)

    return dataset_splits, dataset_builder_info
