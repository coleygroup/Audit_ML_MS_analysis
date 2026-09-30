"""
Sample a train / test assignment for every spectrum from the splitter.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical


@torch.no_grad()
def split_data(dataloader: DataLoader, splitter: torch.nn.Module, train_ratio: float, random_split: bool = False):
    """Iterate over `dataloader` (must NOT shuffle: the returned indices refer to the dataset order) and sample, for
    every spectrum, z = 1 (train) or z = 0 (test).

    With `random_split=True` (first outer loop of learning-to-split) the assignment is Bernoulli(train_ratio);
    otherwise it is sampled from the splitter's softmax output (`splitter.get_output(batch)` -> (batch x 2) logits).

    Returns (split_stats, train_indices, test_indices).
    """
    splitter.eval()
    total_mask = []

    for batch in dataloader:
        if random_split:
            batch_size = len(batch[list(batch.keys())[0]])
            prob = torch.ones(batch_size, 1)
            prob = torch.cat([prob * (1 - train_ratio), prob * train_ratio], dim=-1)   # (test, train)
        else:
            prob = F.softmax(splitter.get_output(batch), dim=-1)

        mask = Categorical(prob).sample().long()   # 0: test, 1: train
        total_mask.append(mask.cpu())

    total_mask = torch.cat(total_mask)
    train_indices = total_mask.nonzero().squeeze(1).tolist()
    test_indices = (1 - total_mask).nonzero().squeeze(1).tolist()
    splitter.train()

    split_stats = {"train_size": len(train_indices),
                   "test_size": len(test_indices),
                   "train_ratio": len(train_indices) / len(total_mask) * 100,
                   "test_ratio": len(test_indices) / len(total_mask) * 100}

    return split_stats, train_indices, test_indices
