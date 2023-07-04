import copy
from typing import Callable, Dict

import wandb
from tqdm.auto import tqdm
from ultralytics.yolo.engine.model import TASK_MAP, YOLO
from ultralytics.yolo.utils import RANK
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.detect.val import DetectionValidator

from .bbox_utils import plot_predictions, plot_validation_results


class WandBUltralyticsCallback:
    def __init__(self, model: YOLO) -> None:
        self.train_validation_table = wandb.Table(
            columns=["Epoch", "Index", "Image", "Mean-Confidence"]
        )
        self.validation_table = wandb.Table(
            columns=["Index", "Image", "Mean-Confidence"]
        )
        self.prediction_table = wandb.Table(
            columns=["Image", "Num-Objects", "Mean-Confidence"]
        )
        self._make_predictor(model)

    def _make_predictor(self, model: YOLO):
        overrides = model.overrides.copy()
        overrides["conf"] = 0.1
        self.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=None)

    def on_fit_epoch_end(self, trainer: DetectionTrainer):
        validator = trainer.validator
        dataloader = validator.dataloader
        class_label_map = validator.names
        self.model = copy.deepcopy(trainer.model)
        self.predictor.setup_model(model=self.model, verbose=False)
        self.train_validation_table = plot_validation_results(
            dataloader=dataloader,
            class_label_map=class_label_map,
            predictor=self.predictor,
            table=self.train_validation_table,
            epoch=trainer.epoch,
        )

    def on_train_end(self, trainer: DetectionTrainer):
        wandb.log({"Train-Validation-Table": self.train_validation_table})

    def on_val_end(self, trainer: DetectionValidator):
        validator = trainer
        dataloader = validator.dataloader
        class_label_map = validator.names
        self.predictor.setup_model(model=self.model, verbose=False)
        self.validation_table = plot_validation_results(
            dataloader=dataloader,
            class_label_map=class_label_map,
            predictor=self.predictor,
            table=self.validation_table,
        )
        wandb.log({"Validation-Table": self.validation_table})

    def on_predict_end(self, predictor: DetectionPredictor):
        for result in tqdm(predictor.results):
            self.prediction_table = plot_predictions(result, self.prediction_table)
        wandb.log({"Prediction-Table": self.prediction_table})

    @property
    def callbacks(self) -> Dict[str, Callable]:
        """Property contains all the relevant callbacks to add to the YOLO model for
        the Weights & Biases logging."""
        return {
            "on_fit_epoch_end": self.on_fit_epoch_end,
            "on_train_end": self.on_train_end,
            "on_val_end": self.on_val_end,
            "on_predict_end": self.on_predict_end,
        }


def add_callback(model: YOLO):
    if RANK in [-1, 0]:
        wandb_callback = WandBUltralyticsCallback(copy.deepcopy(model))
        for event, callback_fn in wandb_callback.callbacks.items():
            model.add_callback(event, callback_fn)
    else:
        wandb.termerror(
            "The RANK of the process to add the callbacks was neither 0 or -1. "
            "No Weights & Biases callbacks were added to this instance of the "
            "YOLO model."
        )
    return model
