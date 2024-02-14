import os
import time

from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_data_loader, DatasetSplit
from model import AlexNet
from train.eval import test_model
from utils import Config


def train(config: Config):
    best_loss = float("inf")
    best_accuracy = 0.

    writer = SummaryWriter(config.result_path)
    model = AlexNet.get_model_from_config(config)
    optimizer = Adam(model.parameters())
    device = model.device

    train_loader = get_data_loader(
        dataset=config.data,
        split=DatasetSplit.TRAIN,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        model.train()

        for inputs, targets in tqdm(train_loader, total=len(train_loader), dynamic_ncols=True,
                                    desc=f"Epoch {epoch}/{config.epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cross_entropy(outputs, targets, reduction="sum")
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if epoch % config.eval_period == 0 or epoch == config.epochs:
            result = test_model(model, writer, config, epoch)
            test_loss = result["loss"]
            test_accuracy = result["accuracy"]

            current_time_in_millis = int(round(time.time() * 1000))
            model_path = os.path.join(config.result_path, f"model_{epoch}_{current_time_in_millis}.pt")

            if test_loss < best_loss and test_accuracy > best_accuracy:
                best_loss = test_loss
                best_accuracy = test_accuracy
                model.save(model_path)
            elif config.save_each_model:
                model.save(model_path)
