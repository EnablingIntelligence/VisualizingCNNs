from enum import Enum
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet

DATASET_ROOT = "./img_data"


class DatasetType(Enum):
    IMAGENET = 1
    CIFAR10 = 2
    CIFAR100 = 3


class DatasetSplit(Enum):
    TRAIN = 1
    TEST = 2


def load_dataset(dataset: DatasetType, split: DatasetSplit, transform: Optional[Callable] = None) -> Dataset:
    """
    When using this function for the ImageNet dataset, make sure to download the dataset first.
    Put the ImageNet data in the folder specified by the DATASET_ROOT variable which defaults to a folder called
    "img_data" in the project root directory.
    Here you can find the instructions: https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html

    All other available datasets are downloaded automatically.

    :param dataset: dataset to load
    :param split: split of the dataset, train or test
    :param transform: optional transform to be applied on a sample
    :return: Dataset
    """
    is_train_data = split == DatasetSplit.TRAIN

    match dataset:
        case DatasetType.IMAGENET:
            split_mode = "train" if is_train_data else "val"
            return ImageNet(root=DATASET_ROOT, split=split_mode, transform=transform)

        case DatasetType.CIFAR10:
            return CIFAR10(root=DATASET_ROOT, train=is_train_data, download=True, transform=transform)

        case DatasetType.CIFAR100:
            return CIFAR100(root=DATASET_ROOT, train=is_train_data, download=True, transform=transform)

    raise ValueError(f"Unsupported dataset {dataset}, supported datasets are: IMAGENET, CIFAR10, CIFAR100")


def get_num_classes(dataset: DatasetType) -> int:
    match dataset:
        case DatasetType.IMAGENET:
            return 1000

        case DatasetType.CIFAR10:
            return 10

        case DatasetType.CIFAR100:
            return 100

    raise ValueError(f"Unsupported dataset {dataset}, supported datasets are: IMAGENET, CIFAR10, CIFAR100")
