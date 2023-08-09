from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow.data as tf_data
import wandb
from keras_core.callbacks import Callback


class ImageClassificationCallback(Callback):
    def __init__(
        self,
        dataset: Union[tf_data.Dataset, Tuple[np.array, np.array]],
        class_labels: Optional[List[str]],
        unbatch_dataset: bool = True,
        max_items_for_visualization: Optional[int] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.class_labels = class_labels
        self.unbatch_dataset = unbatch_dataset
        self.max_items_for_visualization = max_items_for_visualization

        if self.unbatch_dataset:
            self.dataset = self.dataset.unbatch()

        if self.max_items_for_visualization:
            if isinstance(self.dataset, tf_data.Dataset):
                self.dataset = (
                    self.dataset.take(self.max_items_for_visualization)
                    if tf_data.experimental.cardinality(self.dataset).numpy().item()
                    < self.max_items_for_visualization
                    else (
                        self.dataset[0][: self.max_items_for_visualization],
                        self.dataset[1][: self.max_items_for_visualization],
                    )
                )
        else:
            if isinstance(self.dataset, tf_data.Dataset):
                self.max_items_for_visualization = (
                    tf_data.experimental.cardinality(self.dataset).numpy().item()
                )
            else:
                assert self.dataset[0].shape[0] == self.dataset[0].shape[1]
                self.max_items_for_visualization = self.dataset[0].shape[0]

        self.table = wandb.Table(
            columns=[
                "Epoch",
                "Data-Index",
                "Image",
                "Ground-Truth-Label",
                "Predicted-Label",
                "Predicted-Probability",
            ]
        )
