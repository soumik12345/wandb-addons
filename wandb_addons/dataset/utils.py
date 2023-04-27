import os
import shutil
from glob import glob
from pathlib import Path
from packaging import version
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
import wandb
from tensorflow_datasets.core.dataset_builder import DatasetBuilder
from tensorflow_datasets.core.dataset_info import DatasetInfo

from ..utils import upload_wandb_artifact


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


def _upload_tfrecords(
    dataset_name: str, dataset_type: str, aliases: Optional[List[str]] = None
):
    tfrecord_versions_directory = os.path.join(
        os.path.expanduser("~"), "tensorflow_datasets", dataset_name
    )
    tfrecord_versions = sorted(
        [version.parse(v) for v in os.listdir(tfrecord_versions_directory)]
    )
    latest_version_directory = os.path.join(
        tfrecord_versions_directory, str(tfrecord_versions[-1])
    )
    upload_wandb_artifact(
        name=dataset_name,
        artifact_type=dataset_type,
        path=latest_version_directory,
        aliases=aliases,
    )


def _verify_and_create_tfds_module_structure(
    dataset_name: str, dataset_path: str
) -> bool:
    is_tfds_module_structure_valid = False
    builder_script_path = os.path.join(dataset_path, f"{dataset_name}.py")

    if os.path.isfile(builder_script_path):
        wandb.termlog(f"Builder script detected at {dataset_path}")
        # TODO: Create logic to generate TFDS builder module
        wandb.termlog("Attempting to create TFDS Builder module at {dataset_path}")
        tfds_module_path = os.path.join(dataset_path, dataset_name)
        os.makedirs(tfds_module_path)
        _create_empty_file(os.path.join(tfds_module_path, "__init__.py"))
        shutil.move(
            builder_script_path, os.path.join(tfds_module_path, f"{dataset_name}.py")
        )
        is_tfds_module_structure_valid = (
            os.path.isdir(tfds_module_path)
            and os.path.isfile(os.path.join(tfds_module_path, "__init__.py"))
            and os.path.isfile(os.path.join(tfds_module_path, f"{dataset_name}.py"))
        )
    else:
        wandb.termwarn(f"Unable to detect builder script at {dataset_path}")
        tfds_module_path = os.path.join(dataset_path, dataset_name)
        is_tfds_module_structure_valid = (
            os.path.isdir(tfds_module_path)
            and os.path.isfile(os.path.join(tfds_module_path, "__init__.py"))
            and os.path.isfile(os.path.join(tfds_module_path, f"{dataset_name}.py"))
        )

    return is_tfds_module_structure_valid
