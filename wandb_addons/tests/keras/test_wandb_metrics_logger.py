import unittest

import keras_core
import numpy as np
import wandb
from keras_core import layers
from keras_core.utils import to_categorical

from wandb_addons.keras import WandbMetricsLogger


class WandbMetricsLoggerTester(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.run_id = None

    def test_mnist_convnet(self):
        wandb.init(project="wandb-keras-callback-unit-test", entity="geekyrakshit")
        config = wandb.config
        self.run_id = wandb.run.id

        # Model / data parameters
        config.num_classes = 10
        config.input_shape = (28, 28, 1)
        config.batch_size = 128
        config.epochs = 3

        # Load the data and split it between train and test sets
        (x_train, y_train), (x_test, y_test) = keras_core.datasets.mnist.load_data()

        x_train = x_train[:10]
        y_train = y_train[:10]
        x_test = x_test[:10]
        y_test = y_test[:10]

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # convert class vectors to binary class matrices
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

        model.summary()

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