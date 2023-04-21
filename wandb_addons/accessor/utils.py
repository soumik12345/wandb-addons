import os
import shutil
from glob import glob
from pathlib import Path
from typing import Union


def _change_artifact_dir_name(artifact_dir: str):
    new_artifact_dir = artifact_dir.replace(":", "_")
    if os.path.isdir(new_artifact_dir):
        shutil.rmtree(new_artifact_dir)
    os.rename(artifact_dir, new_artifact_dir)
    return new_artifact_dir


def _create_empty_file(filepath: Union[str, Path]):
    with open(filepath, "w") as f:
        pass


def _get_dataset_name_from_artifact_address(artifact_address: str):
    return artifact_address.split(":")[0].split("/")[-1]


def _get_dataset_registration_statement(artifact_dir: str, dataset_name: str):
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
