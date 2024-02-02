from typing import Callable, Optional

from torch.utils.data import DataLoader

from data.dataset import load_dataset, DatasetType, DatasetSplit


def get_data_loader(
        dataset: DatasetType,
        split: DatasetSplit,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 1,
        transform: Optional[Callable] = None
) -> DataLoader:
    return DataLoader(
        dataset=load_dataset(dataset, split, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
