from typing import List, Optional, Tuple, Union

import keras_core as keras
import numpy as np
import tensorflow.data as tf_data
import wandb
from keras_core import backend, ops
from keras_core.callbacks import Callback
from tqdm.auto import tqdm


class WandBImageClassificationCallback(Callback):
    """Callback that logs the images and results of an image-classification task
    including ground-truth and predicted labels and the class-wise probabilities
    in a [wandb.Table](https://docs.wandb.ai/guides/data-vis) in an epoch-wise
    manner.

    !!! example "Example notebooks:"
        - [Image Classification using Keras Core](../examples/image_classification).

    Arguments:
        dataset (Union[tf_data.Dataset, Tuple[np.array, np.array]]): The dataset that
            is to be visualized. This is ideally the validation dataset for the image
            classification task in the form of a
            [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
            that has an `element_spec` like (image_tensor, label_tensor) or a tuple of
            numpy arrays in the form of (image_tensor, label_tensor).
        class_labels (Optional[List[str]]): The list of class names such that the index
            of the class names list corrspond to the labels in the dataset.
        unbatch_dataset (bool) This should be set to `True` if your dataset is batched
            and needs to be unbatched.
        labels_from_logits (bool): Whether the labels in the dataset are
            one-hot-encoded or from logits.
        max_items_for_visualization (Optional[int]): Maximum number of items from the
            dataset to be visualized every epoch.
    """

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
                    else self.dataset
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
            if backend.backend() == "jax":
                predictions = self.model(
                    ops.expand_dims(ops.convert_to_numpy(image), axis=0)
                )
            else:
                predictions = self.model(ops.expand_dims(image, axis=0))
            predicted_label = self.class_labels[
                int(ops.convert_to_numpy(ops.argmax(predictions, axis=-1)).item())
            ]
            predicted_probabilities = self.get_predicted_probabilities(predictions)
            image = ops.convert_to_numpy(image)

            if self.labels_from_logits:
                label = self.class_labels[int(ops.convert_to_numpy(label).item())]
            else:
                if backend.backend() == "jax":
                    label = self.class_labels[
                        int(ops.argmax(ops.convert_to_numpy(label), axis=-1).item())
                    ]
                else:
                    label = self.class_labels[
                        int(ops.convert_to_numpy(ops.argmax(label, axis=-1)).item())
                    ]

            self.table.add_data(
                epoch,
                wandb.Image(image),
                predicted_label,
                label,
                predicted_probabilities,
            )

    def on_train_end(self, logs=None):
        wandb.log({"Evaluation-Table": self.table})
