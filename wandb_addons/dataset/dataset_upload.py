import os
import shlex
import shutil
import subprocess
from typing import List, Optional

import wandb
import tensorflow as tf

from .utils import _create_empty_file
from ..utils import upload_wandb_artifact


def _upload_with_builder_script(name: str, path: str, quiet: bool):
    builder_script_path = os.path.join(path, f"{name}.py")

    if not os.path.isfile(builder_script_path):
        raise wandb.Error(f"Unable to locate builder script {builder_script_path}")
    wandb.termlog(f"Located builder script {builder_script_path}")

    builder_script_module_path = os.path.join(path, name)
    os.makedirs(builder_script_module_path)

    _create_empty_file(os.path.join(builder_script_module_path, "__init__.py"))

    updated_builder_script_file = (
        builder_script_path.split("/")[-1].split(".")[0] + "_dataset_builder.py"
    )
    shutil.move(
        builder_script_path,
        os.path.join(builder_script_module_path, updated_builder_script_file),
    )

    current_working_dir = os.getcwd()
    os.chdir(builder_script_module_path)
    result = subprocess.run(
        shlex.split("tfds build"),
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    os.chdir(current_working_dir)

    if result.returncode != 0:
        wandb.termerror("Unable to build Tensorflow Dataset")
        shutil.move(
            os.path.join(builder_script_module_path, updated_builder_script_file),
            builder_script_path,
        )
        shutil.rmtree(builder_script_module_path)
        subprocess.run(shlex.split("rm -rf ~/tensorflow_datasets/"))
        raise wandb.Error("Unable to build Tensorflow Dataset")

    if not os.path.isfile(os.path.join(path, "__init__.py")):
        _create_empty_file(os.path.join(path, "__init__.py"))


def _upload_with_tfds_directory(name: str, path: str, quiet: bool):
    builder_module_path = os.path.join(path, name)

    if not os.path.isdir(builder_module_path):
        raise wandb.Error(f"Unable to locate tfds module {builder_module_path}")

    current_working_dir = os.getcwd()
    os.chdir(builder_module_path)
    result = subprocess.run(
        shlex.split("tfds build"),
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    os.chdir(current_working_dir)

    if result.returncode != 0:
        wandb.termerror("Unable to build Tensorflow Dataset")
        subprocess.run(shlex.split("rm -rf ~/tensorflow_datasets/"))
        raise wandb.Error("Unable to build Tensorflow Dataset")

    if not os.path.isfile(os.path.join(path, "__init__.py")):
        _create_empty_file(os.path.join(path, "__init__.py"))


def upload_dataset(
    name: str,
    path: str,
    type: str,
    aliases: Optional[List[str]] = None,
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
        name (str): Name of the dataset. This name should follow the
            [PEP8 package and module name convenmtions](https://peps.python.org/pep-0008/#package-and-module-names).
        path (str): Path to the dataset.
        type (str): The type of the artifact, which is used to organize and differentiate
            artifacts. Common typesCinclude dataset or model, but you can use any string containing
            letters, numbers, underscores, hyphens, and dots.
        aliases (Optional[List[str]]): Aliases to apply to this artifact.
        quiet (bool): Whether to suppress the output of dataset build process or not.
    """
    if quiet:
        tf.get_logger().setLevel("ERROR")
    try:
        _upload_with_tfds_directory(name, path, quiet)
        wandb.termlog("Successfully verified tfds module.")
        upload_wandb_artifact(name, type, path, aliases)
    except wandb.Error as e:
        print(e)
        wandb.termlog("Attempting to locate builder script...")
        try:
            _upload_with_builder_script(name, path, quiet)
            wandb.termlog("Successfully verified tfds module.")
            upload_wandb_artifact(name, type, path, aliases)
        except wandb.Error as e:
            print(e)
