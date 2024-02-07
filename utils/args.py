import argparse
import os
import random
import time

import numpy as np
import torch
from torch.backends import cudnn

from data import DatasetType, get_num_classes
from model import NormMode


class Config:
    # pylint: disable=too-many-instance-attributes, too-few-public-methods

    def __init__(self, args):
        self.data = args.data
        self.in_channels = args.in_channels
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.seed = args.seed
        self.train = args.train
        self.model_file = args.model_file
        self.save_each_model = args.save_each_model
        self.eval_period = args.eval_period
        self.run_id = args.run_id
        self.dropout = args.dropout
        self.normalization_method = args.normalization_method
        self.local_size = args.local_size
        self.result_path = args.result_path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Visualizing Convolutional Neural Networks")

    parser.add_argument("--data", type=int, default=1, help="Dataset (1) IMAGENET, (2) CIFAR10, (3) CIFAR100")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use for training and testing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for reproducibility")
    parser.add_argument("--train", type=bool, default=False, help="Train or test the model")
    parser.add_argument("--model_file", type=str, default=None, help="Path to a model file")
    parser.add_argument("--save_each_model", type=bool, default=False, help="Save each model during training")
    parser.add_argument("--eval_period", type=int, default=1, help="Evaluation period during training")
    parser.add_argument("--run_id", type=str, default=f"run_{int(round(time.time() * 1000))}",
                        help="Run ID used for logging")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability used for training")
    parser.add_argument("--normalization_method", type=int, default=1,
                        help="Normalization method (0) Contrast, (1) Local")
    parser.add_argument("--local_size", type=int, default=2, help="Local size for local response normalization")

    return init_config(parser.parse_args())


def init_config(args) -> Config:
    if not args.train and args.model_file is None:
        raise ValueError("Model file must be specified for testing")

    if args.model_file is not None and not os.path.exists(args.model_file):
        raise ValueError(f"Model file {args.model_file} does not exist")

    if args.data not in [1, 2, 3]:
        raise ValueError(f"Invalid dataset {args.data}")

    if args.normalization_method not in [0, 1]:
        raise ValueError(f"Invalid normalization method {args.normalization_method}")

    result_path = os.path.join("result", args.run_id)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    setattr(args, "result_path", result_path)
    setattr(args, "data", DatasetType(args.data))
    setattr(args, "normalization_method", NormMode(args.normalization_method))
    setattr(args, "in_channels", 3)
    setattr(args, "num_classes", get_num_classes(args.data))

    return Config(args)
