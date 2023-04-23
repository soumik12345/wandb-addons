import os
import shlex
import shutil
import subprocess
from typing import Union

import wandb
import tensorflow_datasets as tfds

from .utils import _create_empty_file


def _upload_with_builder_script(name: str, path: str) -> Union[bool, None]:
    builder_script_path = os.path.join(path, f"{name}.py")
    
    if not os.path.isfile(builder_script_path):
        raise wandb.Error(f"Unable to locate builder script {builder_script_path}")
    
    builder_script_module_path = os.path.join(path, name)
    os.makedirs(builder_script_module_path)
    
    _create_empty_file(os.path.join(builder_script_module_path, "__init__.py"))
    
    updated_builder_script_file = builder_script_path.split("/")[-1].split(".")[0] + "_dataset_builder.py"
    wandb.termlog(updated_builder_script_file)
    shutil.move(builder_script_path, os.path.join(builder_script_module_path, updated_builder_script_file))
    
    current_working_dir = os.getcwd()
    os.chdir(builder_script_module_path)
    wandb.termlog(os.getcwd())
    result = subprocess.run(shlex.split("tfds build"))
    os.chdir(current_working_dir)
    
    if result.returncode != 0:
        wandb.termerror("Unable to build Tensorflow Dataset")
        shutil.move(os.path.join(builder_script_module_path, updated_builder_script_file), builder_script_path)
        shutil.rmtree(builder_script_module_path)
        subprocess.run(shlex.split("rm -rf ~/tensorflow_datasets/"))
        raise wandb.Error("Unable to build Tensorflow Dataset")
    
    if not os.path.isfile(os.path.join(path, "__init__.py")):
        _create_empty_file(os.path.join(path, "__init__.py"))


def upload_dataset(name: str, path: str):
    try:
        _upload_with_builder_script(name, path)
    except wandb.Error as e:
        wandb.termerror(e)
