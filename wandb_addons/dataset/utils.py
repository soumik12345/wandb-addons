import os
import shutil
from glob import glob
from pathlib import Path
from typing import Dict, Tuple, Union

import tensorflow as tf
from tensorflow_datasets.core.dataset_info import DatasetInfo
from tensorflow_datasets.core.dataset_builder import DatasetBuilder

import wandb


def _change_artifact_dir_name(artifact_dir: str) -> str:
    new_artifact_dir = artifact_dir.replace(":", "_")
    if os.path.isdir(new_artifact_dir):
        shutil.rmtree(new_artifact_dir)
    os.rename(artifact_dir, new_artifact_dir)
    return new_artifact_dir


def _create_empty_file(filepath: Union[str, Path]):
    with open(filepath, "w") as f:
        pass


def _get_dataset_name_from_artifact_address(artifact_address: str) -> str:
    return artifact_address.split(":")[0].split("/")[-1]


def _get_dataset_registration_statement(artifact_dir: str, dataset_name: str) -> str:
    artifact_dir_import_statement = ".".join(
        [component for component in artifact_dir.split("/")[1:]]
    )
    artifact_dir_import_statement = f"{artifact_dir_import_statement}.{dataset_name}"
    return "import " + artifact_dir_import_statement.strip()


def _remove_redundant_files(artifact_dir: str, dataset_name: str):
    for path in glob(os.path.join(artifact_dir, "*")):
        if path.split("/")[-1] != dataset_name:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isdir(path):
                os.remove(path)


def _build_datasets(
    dataset_builder: DatasetBuilder,
) -> Tuple[Dict[str, tf.data.Dataset], DatasetInfo]:
    dataset_builder_info = dataset_builder.info
    splits = dataset_builder_info.splits
    dataset_splits = {}
    for key, value in splits.items():
        num_shards = dataset_builder.info.splits[key].num_shards
        num_examples = dataset_builder.info.splits[key].num_examples
        wandb.termlog(f"Building dataset for split: {key}...")
        dataset_splits[key] = dataset_builder.as_dataset(key)
        wandb.termlog(
            f"Built dataset for split: {key}, num_shards: {num_shards}, num_examples: {num_examples}"
        )
    return dataset_splits, dataset_builder_info
