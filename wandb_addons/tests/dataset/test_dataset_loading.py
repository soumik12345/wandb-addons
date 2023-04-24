import unittest

from wandb_addons.dataset import load_dataset


class DatasetLoadingFromTFDSModuleTestCase(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.datasets, self.dataset_builder_info = load_dataset(
            "geekyrakshit/monkey-dataset/monkey_species:v2", quiet=True
        )

    def test_classes(self) -> None:
        class_names = self.dataset_builder_info.features["label"].names
        self.assertEqual(len(class_names), 10)

    def test_train_split(self) -> None:
        self.assertEqual(
            tuple(self.datasets["train"].element_spec["image"].shape), (None, None, 3)
        )
        self.assertEqual(tuple(self.datasets["train"].element_spec["label"].shape), ())

    def test_val_split(self) -> None:
        self.assertEqual(
            tuple(self.datasets["val"].element_spec["image"].shape), (None, None, 3)
        )
        self.assertEqual(tuple(self.datasets["val"].element_spec["label"].shape), ())


class DatasetLoadingFromTFRecordsTestCase(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.datasets, self.dataset_builder_info = load_dataset(
            "geekyrakshit/monkey-dataset/monkey_species:v3", quiet=True
        )

    def test_classes(self) -> None:
        class_names = self.dataset_builder_info.features["label"].names
        self.assertEqual(len(class_names), 10)

    def test_train_split(self) -> None:
        self.assertEqual(
            tuple(self.datasets["train"].element_spec["image"].shape), (None, None, 3)
        )
        self.assertEqual(tuple(self.datasets["train"].element_spec["label"].shape), ())

    def test_val_split(self) -> None:
        self.assertEqual(
            tuple(self.datasets["val"].element_spec["image"].shape), (None, None, 3)
        )
        self.assertEqual(tuple(self.datasets["val"].element_spec["label"].shape), ())
