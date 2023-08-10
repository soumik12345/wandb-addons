from typing import Dict, Union

import keras_cv
import numpy as np
import wandb
from keras_core import backend, ops
from tqdm.auto import tqdm


def get_mean_confidence_per_class(
    confidences: Union[backend.KerasTensor, np.array],
    classes: Union[backend.KerasTensor, np.array],
    class_mapping: Dict[int, str],
):
    mean_confidence_dict = {class_name: [] for idx, class_name in class_mapping.items()}
    for idx, confidence in enumerate(confidences):
        class_name = class_mapping[int(classes[idx])]
        mean_confidence_dict[class_name].append(confidence)
    mean_confidence_dict = {
        class_name: sum(confidence_list) / len(confidence_list)
        if len(confidence_list) > 0
        else 0
        for class_name, confidence_list in mean_confidence_dict.items()
    }
    return mean_confidence_dict


def log_predictions_to_wandb(
    image_batch: Union[backend.KerasTensor, np.array],
    prediction_batch: Union[backend.KerasTensor, np.array],
    class_mapping: Dict[int, str],
    source_bbox_format: str = "xywh",
):
    """Function to log inference results to a
    [wandb.Table](https://docs.wandb.ai/guides/data-vis) with images overlayed with an
    interactive bounding box overlay corresponding to the predicted boxes.

    Arguments:
        image_batch (Union[backend.KerasTensor, np.array]): The batch of resized and
            batched images that is also passed to the model.
        prediction_batch (Union[backend.KerasTensor, np.array]): The prediction batch
            that is the output of the detection model.
        class_mapping (Dict[int, str]): A dictionary that maps the index of the classes
            to the corresponding class names.
        source_bbox_format (bool): Format of the source bounding box, one of `"xyxy"`
            or `"xywh"`.
    """
    batch_size = prediction_batch["boxes"].shape[0]
    image_batch = ops.convert_to_numpy(image_batch).astype(np.uint8)
    bounding_boxes = ops.convert_to_numpy(
        keras_cv.bounding_box.convert_format(
            prediction_batch["boxes"],
            source=source_bbox_format,
            target="xyxy",
            images=image_batch,
        )
    )
    table = wandb.Table(columns=["Predictions", "Mean-Confidence"])
    for idx in tqdm(range(batch_size)):
        num_detections = prediction_batch["num_detections"][idx].item()
        predicted_boxes = bounding_boxes[idx][:num_detections]
        confidences = prediction_batch["confidence"][idx][:num_detections]
        classes = prediction_batch["classes"][idx][:num_detections]
        wandb_boxes = []
        for box_idx in range(num_detections):
            wandb_boxes.append(
                {
                    "position": {
                        "minX": predicted_boxes[box_idx][0] / image_batch[idx].shape[0],
                        "minY": predicted_boxes[box_idx][1] / image_batch[idx].shape[1],
                        "maxX": predicted_boxes[box_idx][2] / image_batch[idx].shape[0],
                        "maxY": predicted_boxes[box_idx][3] / image_batch[idx].shape[1],
                    },
                    "class_id": int(classes[box_idx]),
                    "box_caption": class_mapping[int(classes[box_idx])],
                    "scores": {"confidence": float(confidences[box_idx])},
                }
            )
        wandb_image = wandb.Image(
            image_batch[idx],
            boxes={
                "predictions": {
                    "box_data": wandb_boxes,
                    "class_labels": class_mapping,
                },
            },
        )
        mean_confidence_dict = get_mean_confidence_per_class(
            confidences, classes, class_mapping
        )
        table.add_data(wandb_image, mean_confidence_dict)
    wandb.log({"Prediction-Table": table})
