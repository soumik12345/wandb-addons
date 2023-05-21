from typing import Dict, List, Optional

import cv2
import numpy as np
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM

import wandb
from wandb.keras import WandbEvalCallback


_EXPLAINABILITY_MAPPING = {
    "gradcam": {
        "class_name": GradCAM,
        "column_names": ["GradCAM-Heatmap"],
    },
}


class WandbClassificationCallback(WandbEvalCallback):
    """Keras callback for Image Classification Workflow. Logs the images along with the ground-truth labels,
    predicted labels and explainability maps in [Weights & Biases Tables](https://docs.wandb.ai/guides/data-vis)
    for interactive data visualization and analysis.

    **Usage:**

    ```python
    from wandb_addons.keras import WandbExplainabilityCallback

    explainability_callback = WandbExplainabilityCallback(validation_dataset=validation_dataset)
    model.fit(train_dataset, validation_data=validation_dataset, callbacks=[explainability_callback])
    ```

    !!! note "Note"
        Using this callback required a WandB run to be initialized by calling `wandb.init()`.

    Args:
        validation_dataset (tf.data.Dataset): The batched validation dataset.
        data_table_columns (Optional[List[str]]): List of data table columns. By default, it is set to
            `["Index", "Image", "Label"]`.
        pred_table_columns (Optional[List[str]]): List of prediction table columns. By default, it is set to
            `["Epoch", "Idx", "Image", "Label", "Prediction"]`.
        num_samples (int): Maximum number of samples to be visualized.
        id2label (Dict[int, str]): Dictionary mapping the label ids to label names.
        one_hot_label (bool): Whether the labels are one-hot encoded in the `validation_dataset` or not.
    """

    def __init__(
        self,
        validation_dataset,
        explainability_method: str,
        data_table_columns: Optional[List[str]] = None,
        pred_table_columns: Optional[List[str]] = None,
        num_samples: int = 100,
        id2label: dict = None,
        one_hot_label: bool = True,
        **kwargs
    ):
        self.val_data = validation_dataset.unbatch().take(num_samples)
        self.id2label = id2label
        self.one_hot_label = one_hot_label

        data_table_columns = ["Index", "Image", "Ground-Truth"]
        pred_table_columns = [
            "Epoch",
            "Index",
            "Image",
            "Ground-Truth",
            "Predicted-Label",
        ]

        self.alias = explainability_method
        self.explainer = _EXPLAINABILITY_MAPPING[self.alias]["class_name"](**kwargs)
        self.explainer_columns = _EXPLAINABILITY_MAPPING[self.alias]["column_names"]
        pred_table_columns += self.explainer_columns

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

            data = [idx, wandb.Image(image), label]

            self.data_table.add_data(*data)

    def add_model_predictions(self, epoch, logs=None):
        # Get predictions
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()

        explained_results = self._explain()

        for idx in table_idxs:
            pred = preds[idx]

            data = [
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred,
            ]

            for col_name in self.explainer_columns:
                data.append(explained_results[idx][col_name])

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

    def _explain(self):
        explained_results = []
        for image, label in self.val_data:
            image = tf.expand_dims(image, axis=0).numpy()
            label = np.argmax(label) if self.one_hot_label else label.numpy()

            if self.explainer.alias == "gradcam":
                heatmap = self.explainer_entity.explain(
                    validation_data=(image, label),
                    model=self.model,
                    class_index=label,
                    layer_name=None,
                    use_guided_grads=True,
                    colormap=cv2.COLORMAP_VIRIDIS,
                    image_weight=0.7,
                )
                explained_results.append({"GradCAM-Heatmap": wandb.Image(heatmap)})

        return explained_results
