from typing import Dict

import keras_cv
import numpy as np
import tensorflow as tf
import wandb
from keras_core import ops
from tqdm.auto import tqdm


def visualize_dataset(
    dataset,
    class_mapping: Dict[int, str],
    title: str,
    max_batches_to_visualize: int = 1,
    source_bbox_format: str = "xywh",
):
    dataset = dataset.take(max_batches_to_visualize)
    table = wandb.Table(columns=["Images", "Number-of-Objects"])

    for _ in tqdm(range(max_batches_to_visualize)):
        batched_items = next(iter(dataset))
        image_batch, ground_truth_boxes = (
            batched_items["images"],
            batched_items["bounding_boxes"],
        )

        image_batch = keras_cv.utils.to_numpy(image_batch)
        for key, val in ground_truth_boxes.items():
            ground_truth_boxes[key] = keras_cv.utils.to_numpy(val)

        image_batch = ops.convert_to_numpy(image_batch).astype(np.uint8)
        bounding_boxes = ops.convert_to_numpy(
            keras_cv.bounding_box.convert_format(
                ground_truth_boxes["boxes"],
                source=source_bbox_format,
                target="xyxy",
                images=image_batch,
            )
        )

        batch_size = bounding_boxes.shape[0]
        table = wandb.Table(columns=["Images", "Number-of-Objects"])

        for idx in tqdm(range(batch_size)):
            gt_boxes = bounding_boxes[idx]
            classes = ground_truth_boxes["classes"][idx]
            classes = classes[: classes.index(-1)]
            wandb_boxes = []
            num_objects = classes.shape[0]
            for box_idx in range(num_objects):
                wandb_boxes.append(
                    {
                        "position": {
                            "minX": gt_boxes[box_idx][0] / image_batch[idx].shape[0],
                            "minY": gt_boxes[box_idx][1] / image_batch[idx].shape[1],
                            "maxX": gt_boxes[box_idx][2] / image_batch[idx].shape[0],
                            "maxY": gt_boxes[box_idx][3] / image_batch[idx].shape[1],
                        },
                        "class_id": int(classes[box_idx]),
                        "box_caption": class_mapping[int(classes[box_idx])],
                    }
                )
            wandb_image = wandb.Image(
                image_batch[idx],
                boxes={
                    "gorund-truth": {
                        "box_data": wandb_boxes,
                        "class_labels": class_mapping,
                    },
                },
            )
            table.add_data(wandb_image, num_objects)

    wandb.log({title: table})
