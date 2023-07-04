from typing import Callable, Dict

import wandb
from tqdm.auto import tqdm
from ultralytics.yolo.utils import RANK
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.v8.detect.val import DetectionValidator
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

from .bbox_utils import plot_predictions, plot_validation_results


class WandBUltralyticsCallback:
    def __init__(self) -> None:
        self.validation_table = wandb.Table(columns=["Index", "Image"])
        self.prediction_table = wandb.Table(
            columns=["Image", "Num-Objects", "Mean-Confidence"]
        )

    def on_val_end(self, trainer: DetectionValidator):
        validator = trainer
        dataloader = validator.dataloader
        class_label_map = validator.names
        plot_validation_results(dataloader, class_label_map, self.validation_table)

    def on_predict_end(self, predictor: DetectionPredictor):
        for result in tqdm(predictor.results):
            self.prediction_table = plot_predictions(result, self.prediction_table)
        wandb.log({"Prediction-Table": self.prediction_table})

    @property
    def callbacks(self) -> Dict[str, Callable]:
        """Property contains all the relevant callbacks to add to the YOLO model for the Weights & Biases logging."""
        return {"on_val_end": self.on_val_end, "on_predict_end": self.on_predict_end}


def add_callback(model: YOLO):
    if RANK in [-1, 0]:
        wandb_callback = WandBUltralyticsCallback()
        for event, callback_fn in wandb_callback.callbacks.items():
            model.add_callback(event, callback_fn)
    else:
        wandb.termerror(
            "The RANK of the process to add the callbacks was neither 0 or -1."
            "No Weights & Biases callbacks were added to this instance of the YOLO model."
        )
    return model
