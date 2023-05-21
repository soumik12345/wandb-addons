from typing import Dict, List

import numpy as np
import tensorflow as tf

import wandb
from wandb.keras import WandbEvalCallback


class WandbClassificationCallback(WandbEvalCallback):
    def __init__(
        self,
        validation_dataset: tf.data.Dataset,
        data_table_columns: List[str] = ["Index", "Image", "Label"],
        pred_table_columns: List[str] = [
            "Epoch",
            "Idx",
            "Image",
            "Label",
            "Prediction",
        ],
        num_samples: int = 100,
        id2label: Dict[int, str] = None,
        one_hot_label: bool = True,
    ):
        """Keras callback for Image Classification Workflow. Logs the images along with the ground-truth
        and predicted label in Weights & Biases Tables for interactive data visualization and analysis.

        **Usage:**

        ```python
        from wandb_addons.keras import WandbClassificationCallback

        classification_callback = WandbClassificationCallback(validation_dataset=validation_dataset)
        model.fit(train_dataset, validation_data=validation_dataset, callbacks=[classification_callback])
        ```

        !!! note "Note"
            Using this callback required a WandB run to be initialized by calling `wandb.init()`.

        Args:
            validation_dataset (tf.data.Dataset): The batched validation dataset.
            data_table_columns (List[str]): List of data table columns.
            pred_table_columns (List[str]): List of prediction table columns.
            num_samples (int): Maximum number of samples to be visualized.
            id2label (Dict[int, str]): Dictionary mapping the label ids to label names.
            one_hot_label (bool): Whether the labels are one-hot encoded in the `validation_dataset` or not.
        """
        self.val_data = validation_dataset.unbatch().take(num_samples)
        self.id2label = id2label
        self.one_hot_label = one_hot_label

        super().__init__(data_table_columns, pred_table_columns)

    def add_ground_truth(self, logs=None):
        for idx, (image, label) in enumerate(self.val_data):
            if self.one_hot_label:
                label = np.argmax(label, axis=-1)
            else:
                label = label.numpy()

            if self.id2label is not None:
                label = self.id2label.get(label, None)
                # TODO: Add warning if label is None

            data = [
                idx,
                wandb.Image(image),
                label,
            ]

            self.data_table.add_data(*data)

    def add_model_predictions(self, epoch, logs=None):
        # Get predictions
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]

            data = [
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred,
            ]

            self.pred_table.add_data(*data)

    def _inference(self):
        preds = []
        for image, _ in self.val_data:
            pred = self.model(tf.expand_dims(image, axis=0))
            argmax_pred = tf.argmax(pred, axis=-1).numpy()[0]

            if self.id2label is not None:
                argmax_pred = self.id2label.get(argmax_pred, None)

            preds.append(argmax_pred)

        return preds
