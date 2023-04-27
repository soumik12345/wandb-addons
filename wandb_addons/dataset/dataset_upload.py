import os
import shlex
import shutil
import subprocess
from packaging import version
from typing import List, Optional

import wandb
import tensorflow as tf

from .utils import _verify_and_create_tfds_module_structure


def upload_dataset(
    dataset_name: str,
    dataset_path: str,
    aliases: Optional[List[str]] = None,
    upload_tfrecords: bool = True,
    quiet: bool = False,
):
    """Upload and register a dataset with a TFDS module or a TFDS builder script as a
    Weights & Biases artifact. This function would verify if a TFDS build/registration is possible
    with the current specified dataset path and upload it as a Weights & Biases artifact.

    !!! example "Check this guide for preparing a dataset for registering in on Weights & Biases"
        - [Preparing the Dataset](../dataset_preparation).

    Usage:

    ```python
    import wandb
    from wandb_addons.dataset import upload_dataset

    # Initialize a W&B Run
    wandb.init(project="my-awesome-project", job_type="upload_dataset")

    # Note that we should set our dataset name as the name of the artifact
    upload_dataset(name="my_awesome_dataset", path="./my/dataset/path", type="dataset")
    ```

    Args:
        dataset_name (str): Name of the dataset. This name should follow the
            [PEP8 package and module name convenmtions](https://peps.python.org/pep-0008/#package-and-module-names).
        dataset_path (str): Path to the dataset.
        aliases (Optional[List[str]]): Aliases to apply to this artifact.
        upload_tfrecords (bool): Upload dataset as TFRecords or not. If set to `False`, then the dataset is uploaded
            with a TFDS module.
        quiet (bool): Whether to suppress the output of dataset build process or not.
    """
    is_tfds_module_structure_valid = _verify_and_create_tfds_module_structure(
        dataset_name, dataset_path
    )
    if not is_tfds_module_structure_valid:
        wandb.termerror(
            f"Unable to generate or detect valid TFDS module at {dataset_path}"
        )
    else:
        wandb.termlog(f"Verified TFDS module at {dataset_path}")
