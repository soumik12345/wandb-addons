import numpy as np
import tensorflow as tf

import wandb
from wandb.keras import WandbEvalCallback


class WandbClassificationCallback(WandbEvalCallback):
    def __init__(
        self,
        validloader,
        data_table_columns: list,
        pred_table_columns: list,
        num_samples: int = 100,
        id2label: dict = None,
        one_hot_label: bool = True,
    ):
        self.val_data = validloader.unbatch().take(num_samples)
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
