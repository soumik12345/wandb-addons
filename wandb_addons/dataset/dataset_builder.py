from typing import Any, Mapping, Optional, Union

import wandb
import wandb.apis.reports as wr

from etils import epath
import tensorflow_datasets as tfds

from .table_creation import TableCreator
from ..utils import flatten_nested_dictionaries


class WandbDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """An abstract class for Dataset builder that enables building a dataset and upload it as a
    [Weights & Biases Artifact](https://docs.wandb.ai/guides/artifacts). It expects subclasses
    to override the following functions:

    - **`_split_generators`** to return a dict of splits, generators.

    - **`_generate_examples`** to return a generator or an iterator corresponding to the split.

    !!! note "Note"
        Note that this process is alternative to the dataset preparation process using tfds module
        described [here](../dataset_preparation). The dataset registered and uploaded using both
        approaches is easily consumable using the fuction
        [`load_dataset`](./#wandb_addons.dataset.dataset_loading.load_dataset).

    !!! example "Example Artifacts"
        - [🐒 Monkey Dataset](https://wandb.ai/geekyrakshit/artifact-accessor/artifacts/dataset/monkey_dataset).

    ??? example "Example Report"
        <iframe src="https://wandb.ai/geekyrakshit/artifact-accessor/reports/Dataset-monkey-dataset--Vmlldzo0MjgxNTAz" style="border:none;height:1024px;width:100%">

    **Usage:**

    ```python
    import os
    from glob import glob
    from typing import Any, Mapping, Optional, Union

    from etils import epath
    import tensorflow_datasets as tfds

    import wandb
    from wandb_addons.dataset import WandbDatasetBuilder


    class MonkeyDatasetBuilder(WandbDatasetBuilder):
        def __init__(
            self,
            *,
            name: str,
            dataset_path: str,
            features: tfds.features.FeatureConnector,
            upload_raw_dataset: bool = True,
            config: Union[None, str, tfds.core.BuilderConfig] = None,
            data_dir: Optional[epath.PathLike] = None,
            description: Optional[str] = None,
            release_notes: Optional[Mapping[str, str]] = None,
            homepage: Optional[str] = None,
            file_format: Optional[Union[str, tfds.core.FileFormat]] = None,
            disable_shuffling: Optional[bool] = False,
            **kwargs: Any,
        ):
            super().__init__(
                name=name,
                dataset_path=dataset_path,
                features=features,
                upload_raw_dataset=upload_raw_dataset,
                config=config,
                description=description,
                data_dir=data_dir,
                release_notes=release_notes,
                homepage=homepage,
                file_format=file_format,
                disable_shuffling=disable_shuffling,
            )

        def _split_generators(self, dl_manager: tfds.download.DownloadManager):
            return {
                "train": self._generate_examples(
                    os.path.join(self.dataset_path, "training", "training")
                ),
                "val": self._generate_examples(
                    os.path.join(self.dataset_path, "validation", "validation")
                ),
            }

        def _generate_examples(self, path):
            image_paths = glob(os.path.join(path, "*", "*.jpg"))
            for image_path in image_paths:
                label = _CLASS_LABELS[int(image_path.split("/")[-2][-1])]
                yield image_path, {
                    "image": image_path,
                    "label": label,
                }


    if __name__ == "__main__":
        wandb.init(project="artifact-accessor", entity="geekyrakshit")

        builder = MonkeyDatasetBuilder(
            name="monkey_dataset",
            dataset_path="path/to/my/datase",
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "label": tfds.features.ClassLabel(names=_CLASS_LABELS),
                }
            ),
            data_dir="build_dir/",
            description=_DESCRIPTION,
        )

        builder.build_and_upload(create_visualizations=True)
    ```

    Args:
        name (str): A human-readable name for this artifact, which is how you can identify this
            artifact in the UI or reference it in
            [`use_artifact`](https://docs.wandb.ai/ref/python/run#use_artifact) calls. Names can
            contain letters, numbers, underscores, hyphens, and dots. The name must be unique
            across a project.
        dataset_path (str): Path to the dataset.
        features (tfds.features.FeatureConnector): The dataset feature types. Refer to the
            [`tfds.features`](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/)
            module for more information.
        upload_raw_dataset (Optional[bool]): Whether to upload the raw dataset to Weights & Biases
            artifacts as well or not. If set to `True`, the dataset builder would upload the raw
            dataset besides the built dataset, as different versions of the same artifact; with the
            raw dataset being the lower version.
        config (Union[None, str, tfds.core.BuilderConfig]): Dataset configuration.
        data_dir (Optional[epath.PathLike]): The directory where the dataset will be built.
        description (Optional[str]): Description of the dataset as a valid markdown string.
        release_notes (Optional[Mapping[str, str]]): Release notes.
        homepage (Optional[str]): Homepage of the dataset.
        file_format (Optional[Union[str, tfds.core.FileFormat]]): **EXPERIMENTAL**, may change at any
            time; Format of the record files in which dataset will be read/written to. If `None`,
            defaults to `tfrecord`.
        disable_shuffling (Optional[bool]): Disable shuffling of the dataset order.
    """

    def __init__(
        self,
        *,
        name: str,
        dataset_path: str,
        features: tfds.features.FeatureConnector,
        upload_raw_dataset: Optional[bool] = False,
        config: Union[None, str, tfds.core.BuilderConfig] = None,
        data_dir: Optional[epath.PathLike] = None,
        description: Optional[str] = None,
        release_notes: Optional[Mapping[str, str]] = None,
        homepage: Optional[str] = None,
        file_format: Optional[Union[str, tfds.core.FileFormat]] = None,
        disable_shuffling: Optional[bool] = True,
        **kwargs: Any,
    ):
        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before instantiating a subclass of `WandbDatasetBuilder`"
            )

        self.name = name
        self.dataset_path = dataset_path
        self.upload_raw_dataset = upload_raw_dataset
        self.VERSION = self._get_version()
        self.RELEASE_NOTES = release_notes
        if config:
            if isinstance(config, str):
                config = tfds.core.BuilderConfig(
                    name=config, version=self.VERSION, release_notes=release_notes
                )
            self.BUILDER_CONFIGS = [config]
        self._feature_spec = features
        self._description = (
            description or "Dataset built without a DatasetBuilder class."
        )
        self._homepage = homepage
        self._disable_shuffling = disable_shuffling

        self._initialize_wandb_artifact()

        super().__init__(
            data_dir=data_dir,
            config=config,
            version=self.VERSION,
            file_format=file_format,
            **kwargs,
        )

    def _initialize_wandb_artifact(self):
        metadata = {
            "description": self._description,
            "release-notes": self.RELEASE_NOTES,
            "homepage": self._homepage,
        }
        if self.upload_raw_dataset:
            self._wandb_raw_artifact = wandb.Artifact(
                name=self.name,
                type="dataset",
                description=self._description,
                metadata=metadata,
            )
        self._wandb_build_artifact = wandb.Artifact(
            name=self.name,
            type="dataset",
            description=self._description,
            metadata=metadata,
        )

    def _get_version(self) -> tfds.core.utils.Version:
        try:
            api = wandb.Api()
            versions = api.artifact_versions(
                type_name="dataset",
                name=f"{wandb.run.entity}/{wandb.run.project}/{self.name}",
            )
            version = int(next(versions).source_version[1:])
            version = version + 1 if self.upload_raw_dataset else version
            return str(version) + ".0.0"
        except wandb.errors.CommError:
            version = 1 if self.upload_raw_dataset else 0
            return tfds.core.utils.Version(str(version) + ".0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=self._description,
            features=self._feature_spec,
            homepage=self._homepage,
            disable_shuffling=self._disable_shuffling,
        )

    def _create_report(self):
        report = wr.Report(project=wandb.run.project)

        dataset_splits = flatten_nested_dictionaries(self.info.splits).keys()
        scalar_panels = [
            wr.ScalarChart(
                title=f"{split}/num_examples", metric=f"{split}/num_examples"
            )
            for split in dataset_splits
        ]
        scalar_panels += [
            wr.ScalarChart(title=f"{split}/num_shards", metric=f"{split}/num_shards")
            for split in dataset_splits
        ]

        report.title = f"Dataset: {self.name}"

        report.blocks = [
            wr.MarkdownBlock(
                text=f"**Disclaimer:** This report was generated automatically."
            ),
            wr.H1("Description"),
            wr.MarkdownBlock(text=self._description),
            wr.MarkdownBlock("\n"),
        ]

        if self._homepage is not None:
            report.blocks += [
                wr.MarkdownBlock(text=f"**Homepage:** {self._homepage}"),
                wr.MarkdownBlock("\n"),
            ]

        report.blocks += [
            wr.MarkdownBlock(
                f"""
            ```python
            import from wandb_addons.dataset import load_dataset

            datasets, dataset_builder_info = load_dataset("{wandb.run.entity}/{wandb.run.project}/{self.name}:tfrecord")
            ```
            """
            ),
            wr.MarkdownBlock("\n"),
        ]

        report.blocks += [
            wr.PanelGrid(
                runsets=[wr.Runset(project=wandb.run.project, entity=wandb.run.entity)],
                panels=scalar_panels
                + [
                    wr.WeavePanelSummaryTable(table_name=f"{self.name}-Table"),
                    wr.WeavePanelArtifact(self.name),
                ],
            )
        ]

        report.save()

    def build_and_upload(
        self,
        create_visualizations: bool = False,
        max_visualizations_per_split: Optional[int] = None,
    ):
        """Build and prepare the dataset for loading and uploads as a
        [Weights & Biases Artifact](https://docs.wandb.ai/guides/artifacts). This function also
        creates a Weights & Biases reports that contains the dataset description, visualizations
        and all additional metadata logged to Weights & Biases.

        !!! example "Sample Auto-generated Report"
            [🐒 Dataset: monkey-dataset](https://wandb.ai/geekyrakshit/artifact-accessor/reports/Dataset-monkey-dataset--Vmlldzo0MjgxNTAz)
        Args:
            create_visualizations (bool): Automatically parse the dataset and visualize using a
                [Weights & Biase Table](https://docs.wandb.ai/guides/data-vis).
            max_visualizations_per_split (Optional[int]): Maximum number of visualizations per
                split to be visualized in WandB Table. By default, the whole dataset is visualized.
        """
        super().download_and_prepare()

        if create_visualizations:
            table_creator = TableCreator(
                dataset_builder=self,
                dataset_info=self.info,
                max_visualizations_per_split=max_visualizations_per_split,
            )
            table_creator.populate_table()
            table_creator.log(dataset_name=self.name)

        if self.upload_raw_dataset:
            self._wandb_raw_artifact.add_dir(self.dataset_path)
            wandb.log_artifact(self._wandb_raw_artifact, aliases=["raw"])
        self._wandb_build_artifact.add_dir(self.data_dir)
        wandb.log_artifact(self._wandb_build_artifact, aliases=["tfrecord"])

        self._create_report()
