import unittest

import keras_core
import numpy as np
import wandb
from keras_core import layers
from keras_core.utils import to_categorical

from wandb_addons.keras import WandbMetricsLogger


def _test_run(run_id):
    api = wandb.Api()
    run = api.run(f"geekyrakshit/wandb-keras-callback-unit-test/{run_id}")
    config = run.config
    history = run.history()
    run.delete(delete_artifacts=True)
    return config, history


class KerasCallbackTester(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.run_id = None

    def test_mnist_convnet(self):
        wandb.init(project="wandb-keras-callback-unit-test", entity="geekyrakshit")
        config = wandb.config
        self.run_id = wandb.run.id
        config.num_classes = 10
        config.input_shape = (28, 28, 1)
        config.batch_size = 128
        config.epochs = 3
        (x_train, y_train), (x_test, y_test) = keras_core.datasets.mnist.load_data()
        x_train = x_train[:10]
        y_train = y_train[:10]
        x_test = x_test[:10]
        y_test = y_test[:10]
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        y_train = to_categorical(y_train, config.num_classes)
        y_test = to_categorical(y_test, config.num_classes)
        model = keras_core.Sequential(
            [
                layers.Input(shape=config.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(config.num_classes, activation="softmax"),
            ]
        )
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        model.fit(
            x_train,
            y_train,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_split=0.1,
            callbacks=[WandbMetricsLogger(log_freq="batch")],
        )
        wandb.finish()
        config, epoch_history = _test_run(self.run_id)
        self.assertTrue("epochs" in config)
        self.assertTrue("batch_size" in config)
        self.assertTrue("input_shape" in config)
        self.assertTrue("num_classes" in config)
        self.assertEqual(len(epoch_history), 3)
        for history in epoch_history:
            self.assertTrue("epoch/val_loss" in config)
            self.assertTrue("epoch/epoch/val_accuracy" in config)
            self.assertTrue("_runtime" in config)
            self.assertTrue("_timestamp" in config)
            self.assertTrue("epoch/accuracy" in config)
            self.assertTrue("batch/batch_step" in config)
            self.assertTrue("batch/learning_rate" in config)
            self.assertTrue("epoch/learning_rate" in config)
            self.assertTrue("_step" in config)
            self.assertTrue("batch/loss" in config)
            self.assertTrue("epoch/epoch" in config)
            self.assertTrue("batch/accuracy" in config)
            self.assertTrue("epoch/loss" in config)

    def test_mnist_convnet_lr_scheduler(self):
        wandb.init(project="wandb-keras-callback-unit-test", entity="geekyrakshit")
        config = wandb.config
        self.run_id = wandb.run.id
        config.num_classes = 10
        config.input_shape = (28, 28, 1)
        config.batch_size = 128
        config.epochs = 3
        (x_train, y_train), (x_test, y_test) = keras_core.datasets.mnist.load_data()
        x_train = x_train[:10]
        y_train = y_train[:10]
        x_test = x_test[:10]
        y_test = y_test[:10]
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        y_train = to_categorical(y_train, config.num_classes)
        y_test = to_categorical(y_test, config.num_classes)
        model = keras_core.Sequential(
            [
                layers.Input(shape=config.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(config.num_classes, activation="softmax"),
            ]
        )
        lr_schedule = keras_core.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1e-3,
            decay_steps=300,
            end_learning_rate=1e-8,
            power=0.99,
        )
        optimizer = keras_core.optimizers.Adam(
            learning_rate=lr_schedule, weight_decay=0.99
        )
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
        model.fit(
            x_train,
            y_train,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_split=0.1,
            callbacks=[WandbMetricsLogger(log_freq="batch")],
        )
        wandb.finish()
        config, epoch_history = _test_run(self.run_id)
        self.assertTrue("epochs" in config)
        self.assertTrue("batch_size" in config)
        self.assertTrue("input_shape" in config)
        self.assertTrue("num_classes" in config)
        self.assertEqual(len(epoch_history), 3)
        for history in epoch_history:
            self.assertTrue("epoch/val_loss" in config)
            self.assertTrue("epoch/epoch/val_accuracy" in config)
            self.assertTrue("_runtime" in config)
            self.assertTrue("_timestamp" in config)
            self.assertTrue("epoch/accuracy" in config)
            self.assertTrue("batch/batch_step" in config)
            self.assertTrue("batch/learning_rate" in config)
            self.assertTrue("epoch/learning_rate" in config)
            self.assertTrue("_step" in config)
            self.assertTrue("batch/loss" in config)
            self.assertTrue("epoch/epoch" in config)
            self.assertTrue("batch/accuracy" in config)
            self.assertTrue("epoch/loss" in config)
