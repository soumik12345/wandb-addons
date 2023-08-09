from typing import List, Optional, Tuple, Union

import keras_core as keras
import numpy as np
import tensorflow.data as tf_data
import wandb
from keras_core import ops
from keras_core.callbacks import Callback
from tqdm.auto import tqdm


class WandBImageClassificationCallback(Callback):
    def __init__(
        self,
        dataset: Union[tf_data.Dataset, Tuple[np.array, np.array]],
        class_labels: Optional[List[str]],
        unbatch_dataset: bool = True,
        labels_from_logits: bool = False,
        max_items_for_visualization: Optional[int] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.class_labels = class_labels
        self.unbatch_dataset = unbatch_dataset
        self.labels_from_logits = labels_from_logits
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
                "Image",
                "Ground-Truth-Label",
                "Predicted-Label",
                "Predicted-Probability",
            ]
        )

    def get_predicted_probabilities(self, predictions: np.array):
        predictions = ops.convert_to_numpy(ops.squeeze(predictions)).tolist()
        return {
            self.class_labels[idx]: predictions[idx] for idx in range(len(predictions))
        }

    def on_epoch_end(self, epoch, logs=None):
        data_iterator = (
            iter(self.dataset)
            if isinstance(self.dataset, tf_data.Dataset)
            else zip(self.dataset)
        )
        data_iterator = tqdm(
            data_iterator,
            total=self.max_items_for_visualization,
            desc="Populating W&B Table",
        )
        for image, label in data_iterator:
            predictions = self.model(ops.expand_dims(image, axis=0))
            predicted_label = self.class_labels[
                int(ops.convert_to_numpy(ops.argmax(predictions, axis=-1)).item())
            ]
            predicted_probabilities = self.get_predicted_probabilities(predictions)
            image = ops.convert_to_numpy(image)
            label = (
                self.class_labels[int(ops.convert_to_numpy(label).item())]
                if self.labels_from_logits
                else self.class_labels[
                    int(ops.convert_to_numpy(ops.argmax(label, axis=-1)).item())
                ]
            )
            self.table.add_data(
                epoch,
                wandb.Image(image),
                predicted_label,
                label,
                predicted_probabilities,
            )

    def on_train_end(self, logs=None):
        wandb.log({"Evaluation-Table": self.table})
