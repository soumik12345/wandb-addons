from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadManager, DownloadMode
from datasets.utils.info_utils import VerificationMode
import wandb

import datasets
from datasets.data_files import DataFilesDict
from datasets.features import Features
from datasets.info import DatasetInfo


class WandbDatasetBuilder(datasets.GeneratorBasedBuilder):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        metadata: dict,
        cache_dir: str | None = None,
        config_name: str | None = None,
        hash: str | None = None,
        base_path: str | None = None,
        info: DatasetInfo | None = None,
        features: Features | None = None,
        use_auth_token: bool | str | None = None,
        repo_id: str | None = None,
        data_files: str | list | dict | DataFilesDict | None = None,
        data_dir: str | None = None,
        storage_options: dict | None = None,
        writer_batch_size: int | None = None,
        name="deprecated",
        **config_kwargs
    ):
        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before instantiating a subclass of `WandbDatasetBuilder`"
            )

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.metadata = metadata

        super().__init__(
            cache_dir,
            config_name,
            hash,
            base_path,
            info,
            features,
            use_auth_token,
            repo_id,
            data_files,
            data_dir,
            storage_options,
            writer_batch_size,
            name,
            **config_kwargs
        )

    def download_and_prepare(
        self,
        output_dir: str | None = None,
        download_config: DownloadConfig | None = None,
        download_mode: DownloadMode | str | None = None,
        verification_mode: VerificationMode | str | None = None,
        ignore_verifications="deprecated",
        try_from_hf_gcs: bool = True,
        dl_manager: DownloadManager | None = None,
        base_path: str | None = None,
        use_auth_token="deprecated",
        file_format: str = "arrow",
        max_shard_size: int | str | None = None,
        num_proc: int | None = None,
        storage_options: dict | None = None,
        **download_and_prepare_kwargs
    ):
        super().download_and_prepare(
            output_dir,
            download_config,
            download_mode,
            verification_mode,
            ignore_verifications,
            try_from_hf_gcs,
            dl_manager,
            base_path,
            use_auth_token,
            file_format,
            max_shard_size,
            num_proc,
            storage_options,
            **download_and_prepare_kwargs
        )

        self.metadata.update({"description": self.config.description})
        self._wandb_build_artifact = wandb.Artifact(
            name=self.dataset_name,
            type="dataset",
            description=self.config.description,
            metadata=self.metadata,
        )

        self._wandb_build_artifact.add_dir(output_dir)
        wandb.log_artifact(self._wandb_build_artifact, aliases=[file_format])
