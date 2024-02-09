from typing import Dict

import torch
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_data_loader, DatasetSplit
from model import AlexNet
from utils import Config


def test(config: Config) -> Dict[str, float]:
    writer = SummaryWriter(config.result_path)
    model = AlexNet.get_model_from_config(config)
    return test_model(model, writer, config)


def test_model(model: AlexNet, writer: SummaryWriter, config: Config, global_step: int = 0) -> Dict[str, float]:
    n_iter = 0
    total_loss = 0.0
    correct_classifications = 0
    test_loader = get_data_loader(
        dataset=config.data,
        split=DatasetSplit.TEST,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    n_samples = len(test_loader.dataset)

    model.eval()
    device = model.device

    with torch.no_grad():
        progress_bar = tqdm(test_loader, initial=1, dynamic_ncols=True)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            n_iter += 1

            outputs = model(inputs)

            loss = cross_entropy(outputs, targets, reduction="sum")
            total_loss += loss.item()

            _, predictions = torch.max(outputs, dim=1)
            correct_classifications += (predictions == targets).sum().item()

            progress_bar.set_description(f"Test Iter: [{n_iter}/{len(test_loader)}]")

    test_loss = total_loss / n_samples
    writer.add_scalar("Loss/test", test_loss, global_step)

    test_accuracy = 100 * correct_classifications / n_samples
    writer.add_scalar("Accuracy/test", test_accuracy, global_step)

    return {"accuracy": test_accuracy, "loss": test_loss}
