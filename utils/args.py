import argparse
import os
import random

import numpy as np
import torch
from torch.backends import cudnn

from data import Data


class Config:
    # pylint: disable=too-many-instance-attributes, too-few-public-methods

    def __init__(self, args):
        self.data = args.data
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.seed = args.seed
        self.train = args.train
        self.model = args.model
        self.save_each_model = args.save_each_model
        self.eval_period = args.eval_period
        self.run_id = args.run_id


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Visualizing Convolutional Neural Networks")

    parser.add_argument("--data", type=Data, default=int,
                        help="Dataset (1) MNIST, (2) CIFAR10, (3) CIFAR100, (4) IMAGENET")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train", type=bool, default=True, help="Train or test the model")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--save_each_model", type=bool, default=False, help="Save each model")
    parser.add_argument("--eval_period", type=int, default=1, help="Evaluation period")
    parser.add_argument("--run_id", type=str, default="0", help="Run ID used for logging")

    return init_config(parser.parse_args())


def init_config(args) -> Config:
    if not args.train and args.model is None:
        raise ValueError("Model file must be specified for testing")

    if args.model is not None and not os.path.exists(args.model):
        raise ValueError(f"Model file {args.model} does not exist")

    if args.data not in [1, 2, 3, 4]:
        raise ValueError(f"Invalid dataset {args.data}")

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
    setattr(args, "data", Data(args.data))

    return Config(args)
