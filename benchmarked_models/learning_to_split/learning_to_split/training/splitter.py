"""
Loss functions used to train a *splitter*: a network that assigns every spectrum a probability of belonging to the
train (z = 1) or the test (z = 0) split. Following "Learning to Split for Automatic Bias Detection" (Bao & Barzilay,
2022), the splitter is trained so that

  1. the predictor makes its mistakes on the spectra sent to the test split (`compute_gap_loss`), and
  2. the expected train fraction matches a target ratio (`compute_marginal_z_loss`).

An optional third term (`compute_y_given_z_loss`) keeps the fingerprint-bit marginals of the two splits equal; the
drivers in this repository do not use it. All functions take the splitter's raw 2-class logits (batch x 2, column 1 =
train) so that the callers (`MistNetSplitter`, `MSBinnedModelSplitter`, ...) never have to apply a softmax themselves.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def jaccard_dist(FP_pred: torch.Tensor, FP: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Per-sample Jaccard (Tanimoto) distance between predicted fingerprint probabilities, binarised at `threshold`,
    and the true binary fingerprints. Both tensors are (batch x n_bits)."""
    FP_pred = (FP_pred > threshold).bool()
    FP = FP.bool()
    intersection = (FP & FP_pred).sum(dim=1)
    union = (FP | FP_pred).sum(dim=1)
    return 1.0 - intersection.float() / (union.float() + 1e-9)


def compute_gap_loss(logits: torch.Tensor, FP_pred: torch.Tensor, FP: torch.Tensor,
                     threshold: float = 0.5, jaccard_threshold: float = 0.75) -> torch.Tensor:
    """Cross-entropy that pushes well-predicted spectra (Jaccard distance <= jaccard_threshold) towards the train
    split and poorly predicted ones towards the test split.

    Args:
        logits: splitter output, (batch x 2); column 0 = test, column 1 = train.
        FP_pred: predictor output probabilities, (batch x n_bits).
        FP: ground-truth fingerprints, (batch x n_bits).
        threshold: binarisation threshold applied to FP_pred.
        jaccard_threshold: distance above which a spectrum counts as a mistake.
    """
    dist = jaccard_dist(FP_pred, FP, threshold)
    target = (dist <= jaccard_threshold).long()      # 1: train, 0: test
    return F.cross_entropy(logits, target)


def compute_marginal_z_loss(logits: torch.Tensor, tar_ratio: float, no_grad: bool = False):
    """KL divergence between the splitter's marginal train probability and the target train ratio.

    Returns (loss, current_train_ratio)."""
    prob = F.softmax(logits, dim=-1)
    cur_ratio = torch.mean(prob[:, 1])
    cur_z = torch.stack([1.0 - cur_ratio, cur_ratio])        # (test, train)
    tar_ratio = torch.ones_like(cur_ratio) * tar_ratio
    tar_z = torch.stack([1.0 - tar_ratio, tar_ratio])

    loss_ratio = F.kl_div(torch.log(cur_z), tar_z, reduction="batchmean")
    if not torch.isfinite(loss_ratio):
        loss_ratio = torch.ones_like(loss_ratio)
    if no_grad:
        loss_ratio = loss_ratio.item()
    return loss_ratio, cur_ratio.item()


def compute_y_given_z_loss(logits: torch.Tensor, FP: torch.Tensor, no_grad: bool = False, eps: float = 1e-6):
    """Symmetric KL between the per-bit fingerprint marginals of the (soft) train and test splits and the marginal of
    the whole batch, i.e. p(y | z = 1) ~ p(y | z = 0) ~ p(y). Not used by the drivers in this repository."""
    w_train = F.softmax(logits, dim=-1)[:, 1]                # (batch,)
    w_test = 1.0 - w_train
    y = (FP == 1).float()                                    # (batch x n_bits)

    p_given_train = (y * w_train[:, None]).sum(0) / w_train.sum()
    p_given_test = (y * w_test[:, None]).sum(0) / w_test.sum()
    p_original = y.mean(0).detach()

    loss = 0.5 * (F.kl_div(torch.log(p_given_train.clamp_min(eps)), p_original, reduction="batchmean")
                  + F.kl_div(torch.log(p_given_test.clamp_min(eps)), p_original, reduction="batchmean"))
    if not torch.isfinite(loss):
        loss = torch.ones_like(loss)
    if no_grad:
        loss = loss.item()
    return loss
