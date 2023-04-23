import os
from collections.abc import MutableMapping
from typing import Dict

import wandb
from wandb.util import FilePathStr


def upload_wandb_artifact(name: str, artifact_type: str, path: str):
    if wandb.run is not None:
        artifact = wandb.Artifact("animals", type="dataset")
        artifact.add_dir("images")  # Adds multiple files to artifact
        wandb.log_artifact(artifact)
    else:
        raise wandb.Error("You must call `wandb.init()` before logging an artifact.")


def fetch_wandb_artifact(artifact_address: str, artifact_type: str) -> FilePathStr:
    """Utility function for fetching a
    [Weights & Biases artifact](https://docs.wandb.ai/guides/artifacts)
    irrespective of whether a [run](https://docs.wandb.ai/guides/runs) has been initialized or not.

    Args:
        artifact_address (str): A human-readable name for the artifact, which is how you can
            identify the artifact in the UI or reference it in
            [`use_artifact`](https://docs.wandb.ai/ref/python/run#use_artifact) calls. Names can
            contain letters, numbers, underscores, hyphens, and dots. The name must be unique across
            a project.
        artifact_type (str): The type of the artifact, which is used to organize and differentiate
            artifacts. Common typesCinclude dataset or model, but you can use any string containing
            letters, numbers, underscores, hyphens, and dots.

    Returns:
        (wandb.util.FilePathStr): The path to the downloaded contents.
    """
    return (
        wandb.Api().artifact(artifact_address, type=artifact_type).download()
        if wandb.run is None
        else wandb.use_artifact(artifact_address, type=artifact_type).download()
    )


def flatten_nested_dictionaries(d: Dict, parent_key: str = "", sep: str = "/") -> Dict:
    """A recursive function for flattening nested dictionaries.

    # Reference:
        Answer to
        [**Flatten nested dictionaries, compressing keys**](https://stackoverflow.com/q/6027558)
        on StackOverflow: [stackoverflow.com/a/6027615](https://stackoverflow.com/a/6027615)

    Args:
        d (Dict): The input nested dictionary.
        parent_key (str): The parent key.
        sep (str): The separator to use for the flattened keys.

    Returns:
        (Dict): The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_nested_dictionaries(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
