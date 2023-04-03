import wandb
from wandb.util import FilePathStr


def fetch_wandb_artifact(artifact_address: str, artifact_type: str) -> FilePathStr:
    """
    Utility function for fetching a [Weights & Biases artifact](https://docs.wandb.ai/guides/artifacts)
    irrespective of whether a [run](https://docs.wandb.ai/guides/runs) has been initialized or not.

    # Arguments:
        artifact_address: str.
            A human-readable name for the artifact, which is how you can identify the artifact in the UI
            or reference it in [`use_artifact`](https://docs.wandb.ai/ref/python/run#use_artifact) calls.
            Names can contain letters, numbers, underscores, hyphens, and dots. The name must be unique
            across a project.
        artifact_type: str.
            The type of the artifact, which is used to organize and differentiate artifacts. Common types
            include dataset or model, but you can use any string containing letters, numbers, underscores,
            hyphens, and dots.

    # Returns:
        (wandb.util.FilePathStr): The path to the downloaded contents.
    """
    return (
        wandb.Api().artifact(artifact_address, type=artifact_type).download()
        if wandb.run is None
        else wandb.use_artifact(artifact_address, type=artifact_type).download()
    )
