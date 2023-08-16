from typing import Optional, Union

import keras_cv
import numpy as np
import wandb
from tensorflow import data as tf_data
from tensorflow import keras
from tqdm.auto import tqdm

from .inference import get_mean_confidence_per_class


class WandBDetectionVisualizationCallback(keras.callbacks.Callback):
    """Callback for visualizing ground-truth and predicted bounding boxes in an
    epoch-wise manner for an object-detection task for
    [KerasCV](https://github.com/keras-team/keras-cv). The callback logs a
    [`wandb.Table`](https://docs.wandb.ai/guides/tables) with columns for the epoch,
    the images overlayed with an interactive bounding box overlay corresponding to the
    ground-truth and predicted boudning boxes, the number of ground-truth bounding
    boxes and the predicted mean-confidence for each class.

    !!! example "Examples:"
        - [Fine-tuning an Object Detection Model using KerasCV](../examples/train_retinanet).
        - [Sample Results for Fine-tuning an Object Detection Model using KerasCV](https://wandb.ai/geekyrakshit/keras-cv-callbacks/reports/Keras-CV-Integration--Vmlldzo1MTU4Nzk3)

    Arguments:
        dataset (tf.data.Dataset): A batched dataset consisting of Ragged Tensors.
            This can be obtained by applying `ragged_batch()` on a `tf.data.Dataset`.
        class_mapping (Dict[int, str]): A dictionary that maps the index of the classes
            to the corresponding class names.
        max_batches_to_visualize (Optional[int]): Maximum number of batches from the
            dataset to be visualized.
        iou_threshold (float): IoU threshold for non-max suppression during prediction.
        confidence_threshold (float): Confidence threshold for non-max suppression
            during prediction.
        source_bbox_format (str): Format of the source bounding box, one of `"xyxy"`
            or `"xywh"`.
        title (str): Title under which the table will be logged to the Weights & Biases
            workspace.
    """

    def __init__(
        self,
        dataset: tf_data.Dataset,
        class_mapping: dict,
        max_batches_to_visualize: Optional[Union[int, None]] = 1,
        iou_threshold: float = 0.01,
        confidence_threshold: float = 0.01,
        source_bounding_box_format: str = "xywh",
        title: str = "Evaluation-Table",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dataset = dataset.take(max_batches_to_visualize)
        self.class_mapping = class_mapping
        self.max_batches_to_visualize = max_batches_to_visualize
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.source_bounding_box_format = source_bounding_box_format
        self.title = title
        self.prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
            bounding_box_format=self.source_bounding_box_format,
            from_logits=True,
            iou_threshold=self.iou_threshold,
            confidence_threshold=self.confidence_threshold,
        )
        self.table = wandb.Table(
            columns=[
                "Epoch",
                "Image",
                "Number-of-Ground-Truth-Boxes",
                "Mean-Confidence",
            ]
        )

    def plot_prediction(self, epoch, image_batch, y_true_batch):
        y_pred_batch = self.model.predict(image_batch)
        y_pred = keras_cv.bounding_box.to_ragged(y_pred_batch)
        image_batch = keras_cv.utils.to_numpy(image_batch).astype(np.uint8)
        ground_truth_bounding_boxes = keras_cv.utils.to_numpy(
            keras_cv.bounding_box.convert_format(
                y_true_batch["boxes"],
                source=self.source_bounding_box_format,
                target="xyxy",
                images=image_batch,
            )
        )
        ground_truth_classes = keras_cv.utils.to_numpy(y_true_batch["classes"])
        predicted_bounding_boxes = keras_cv.utils.to_numpy(
            keras_cv.bounding_box.convert_format(
                y_pred["boxes"],
                source=self.source_bounding_box_format,
                target="xyxy",
                images=image_batch,
            )
        )
        for idx in range(image_batch.shape[0]):
            num_detections = y_pred["num_detections"][idx].item()
            predicted_boxes = predicted_bounding_boxes[idx][:num_detections]
            confidences = keras_cv.utils.to_numpy(
                y_pred["confidence"][idx][:num_detections]
            )
            predicted_classes = keras_cv.utils.to_numpy(
                y_pred["classes"][idx][:num_detections]
            )

            gt_classes = [
                int(class_idx) for class_idx in ground_truth_classes[idx].tolist()
            ]
            gt_boxes = ground_truth_bounding_boxes[idx]
            if -1 in gt_classes:
                gt_classes = gt_classes[: gt_classes.index(-1)]

            wandb_prediction_boxes = []
            for box_idx in range(num_detections):
                wandb_prediction_boxes.append(
                    {
                        "position": {
                            "minX": predicted_boxes[box_idx][0]
                            / image_batch[idx].shape[0],
                            "minY": predicted_boxes[box_idx][1]
                            / image_batch[idx].shape[1],
                            "maxX": predicted_boxes[box_idx][2]
                            / image_batch[idx].shape[0],
                            "maxY": predicted_boxes[box_idx][3]
                            / image_batch[idx].shape[1],
                        },
                        "class_id": int(predicted_classes[box_idx]),
                        "box_caption": self.class_mapping[
                            int(predicted_classes[box_idx])
                        ],
                        "scores": {"confidence": float(confidences[box_idx])},
                    }
                )

            wandb_ground_truth_boxes = []
            for box_idx in range(len(gt_classes)):
                wandb_ground_truth_boxes.append(
                    {
                        "position": {
                            "minX": int(gt_boxes[box_idx][0]),
                            "minY": int(gt_boxes[box_idx][1]),
                            "maxX": int(gt_boxes[box_idx][2]),
                            "maxY": int(gt_boxes[box_idx][3]),
                        },
                        "class_id": gt_classes[box_idx],
                        "box_caption": self.class_mapping[int(gt_classes[box_idx])],
                        "domain": "pixel",
                    }
                )
            wandb_image = wandb.Image(
                image_batch[idx],
                boxes={
                    "ground-truth": {
                        "box_data": wandb_ground_truth_boxes,
                        "class_labels": self.class_mapping,
                    },
                    "predictions": {
                        "box_data": wandb_prediction_boxes,
                        "class_labels": self.class_mapping,
                    },
                },
            )
            mean_confidence_dict = get_mean_confidence_per_class(
                confidences, predicted_classes, self.class_mapping
            )
            self.table.add_data(
                epoch, wandb_image, len(gt_classes), mean_confidence_dict
            )

    def on_epoch_end(self, epoch, logs):
        original_prediction_decoder = self.model._prediction_decoder
        self.model.prediction_decoder = self.prediction_decoder
        for _ in tqdm(range(self.max_batches_to_visualize)):
            image_batch, y_true_batch = next(iter(self.dataset))
            self.plot_prediction(epoch, image_batch, y_true_batch)
        self.model.prediction_decoder = original_prediction_decoder

    def on_train_end(self, logs):
        wandb.log({self.title: self.table})
