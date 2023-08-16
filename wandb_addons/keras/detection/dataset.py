from typing import Dict, Optional

import keras_cv
import wandb
from tensorflow import data as tf_data
from tqdm.auto import tqdm


def visualize_dataset(
    dataset: tf_data.Dataset,
    class_mapping: Dict[int, str],
    title: str,
    max_batches_to_visualize: Optional[int] = 1,
    source_bbox_format: str = "xywh",
):
    """Function to visualize a dataset using a
    [wandb.Table](https://docs.wandb.ai/guides/data-vis) with 2 columns, one with the
    images overlayed with an interactive bounding box overlay corresponding to the
    predicted boxes and another showing the number of bounding boxes corresponding to
    that image.

    !!! example "Example notebooks:"
        - [Object Detection using KerasCV](../examples/visualize_dataset).

    Arguments:
        dataset (tf.data.Dataset): A batched dataset consisting of Ragged Tensors.
        This can be obtained by applying `ragged_batch()` on a `tf.data.Dataset`.
        class_mapping (Dict[int, str]): A dictionary that maps the index of the classes
            to the corresponding class names.
        title (str): Title under which the table will be logged to the Weights & Biases
            workspace.
        max_batches_to_visualize (Optional[int]): Maximum number of batches from the
            dataset to be visualized.
        source_bbox_format (str): Format of the source bounding box, one of `"xyxy"`
            or `"xywh"`.
    """
    table = wandb.Table(columns=["Images", "Number-of-Objects"])
    if max_batches_to_visualize is not None:
        dataset = iter(dataset.take(max_batches_to_visualize))
    else:
        dataset = iter(dataset)
        max_batches_to_visualize = tf_data.experimental.cardinality(dataset).numpy()

    for _ in tqdm(range(max_batches_to_visualize)):
        sample = next(dataset)
        images, bounding_boxes = sample["images"], sample["bounding_boxes"]
        images = keras_cv.utils.to_numpy(images)
        images = keras_cv.utils.transform_value_range(
            images, original_range=(0, 255), target_range=(0, 255)
        )
        for key, val in bounding_boxes.items():
            bounding_boxes[key] = keras_cv.utils.to_numpy(val)
        bounding_boxes["boxes"] = keras_cv.bounding_box.convert_format(
            bounding_boxes["boxes"],
            source=source_bbox_format,
            target="xyxy",
            images=images,
        )
        bounding_boxes["boxes"] = keras_cv.utils.to_numpy(bounding_boxes["boxes"])
        for idx in range(images.shape[0]):
            classes = [
                int(class_idx) for class_idx in bounding_boxes["classes"][idx].tolist()
            ]
            bboxes = bounding_boxes["boxes"][idx]
            if -1 in classes:
                classes = classes[: classes.index(-1)]
            wandb_boxes = []
            for object_idx in range(len(classes)):
                wandb_boxes.append(
                    {
                        "position": {
                            "minX": int(bboxes[object_idx][0]),
                            "minY": int(bboxes[object_idx][1]),
                            "maxX": int(bboxes[object_idx][2]),
                            "maxY": int(bboxes[object_idx][3]),
                        },
                        "class_id": classes[object_idx],
                        "box_caption": class_mapping[int(classes[object_idx])],
                        "domain": "pixel",
                    }
                )
            wandb_image = wandb.Image(
                images[idx],
                boxes={
                    "gorund-truth": {
                        "box_data": wandb_boxes,
                        "class_labels": class_mapping,
                    },
                },
            )
            table.add_data(wandb_image, len(classes))

    wandb.log({title: table})
