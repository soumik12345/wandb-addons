import wandb
from tqdm.auto import tqdm

import numpy as np
import tensorflow_datasets as tfds

from ..utils import flatten_nested_dictionaries


_FEATURE_MAPPING = {
    "ClassLabel": lambda data, feature: feature.names[
        int(data.numpy().item()) if hasattr(feature, "names") else data.numpy().item()
    ],
    "Image": lambda data, feature: wandb.Image(data.numpy().astype(np.uint8)),
}


class TableCreator:
    def __init__(
        self,
        dataset_builder: tfds.core.DatasetBuilder,
        dataset_info: tfds.core.DatasetInfo,
    ):
        self.dataset_builder = dataset_builder
        self.dataset_info = dataset_info
        self.features = flatten_nested_dictionaries(self.dataset_info.features)
        self.splits = flatten_nested_dictionaries(self.dataset_info.splits)
        self._columns = ["Split"] + list(self.features.keys())
        self._table = wandb.Table(columns=self._columns)
        self._feature_mapping = {
            "ClassLabel": lambda x: x.numpy().item(),
            "Image": lambda x: wandb.Image(x.numpy().astype(np.uint8)),
        }

    def populate_table(self):
        wandb.termwarn(
            "Creating visualization tables is not vertically scalable as of this point."
        )
        for split, _ in self.splits.items():
            num_examples, num_shards = (
                self.splits[split].num_examples,
                self.splits[split].num_shards,
            )
            wandb.log(
                {
                    f"{split}/num_examples": num_examples,
                    f"{split}/num_shards": num_shards,
                },
            )
            dataset = iter(self.dataset_builder.as_dataset(split))
            for _ in tqdm(
                range(num_examples), desc=f"Populating Table for the {split} split"
            ):
                data = next(dataset)
                row = [split] + [
                    _FEATURE_MAPPING[type(self.features[col]).__name__](
                        data[col], self.features[col]
                    )
                    for col in self._columns[1:]
                ]
                self._table.add_data(*row)

    def log(self, dataset_name: str):
        wandb.log({f"{dataset_name}-Table": self._table})
