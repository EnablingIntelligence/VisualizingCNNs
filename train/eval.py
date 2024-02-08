import torch
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_data_loader, DatasetSplit
from model import AlexNet
from utils import Config


def test(config: Config):
    writer = SummaryWriter(config.result_path)
    model = AlexNet.get_model_from_config(config)
    test_model(model, writer, config)


def test_model(model: AlexNet, writer: SummaryWriter, config: Config) -> float:
    total_loss = 0.0
    n_iter = 0
    test_loader = get_data_loader(
        dataset=config.data,
        split=DatasetSplit.TEST,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    model.eval()
    device = model.device

    with torch.no_grad():
        progress_bar = tqdm(test_loader, initial=1, dynamic_ncols=True)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = cross_entropy(outputs, targets)

            total_loss += loss.item()
            n_iter += 1
            avg_loss = total_loss / n_iter

            progress_bar.set_description(f"Test Iter: [{n_iter}/{len(test_loader)}] Loss: {avg_loss:.4f}")
            writer.add_scalar("Loss/test", avg_loss, n_iter)

    return avg_loss
