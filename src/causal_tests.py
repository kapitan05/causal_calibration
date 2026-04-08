from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import auc


@torch.no_grad()
def evaluate_causal_metric(
    model: torch.nn.Module,
    sequence_tensor: torch.Tensor,
    perturbation_levels: list[float],
    target_class: int,
    batch_size: int = 32,
) -> tuple[list[float], float, npt.NDArray[np.float64]]:
    """
    Evaluates a pre-generated sequence of images and calculates the AUC score.

    Args:
        model: PyTorch model in evaluation mode.
        sequence_tensor: Tensor, shape: [num_steps, C, H, W] containing modified images.
        perturbation_levels: Exact fraction of pixels modified at each step.
        target_class: The index of the class being explained.
        batch_size: Number of images to process simultaneously for speedup.

    Returns:
        List of probabilities at each step, and the AUC score.
    """

    num_steps = sequence_tensor.size(0)
    probabilities: list[float] = []
    all_probs: list[npt.NDArray[np.float64]] = []

    # model evaluation of sequence using batches
    for i in range(0, num_steps, batch_size):
        batch = sequence_tensor[i : i + batch_size]
        batch_levels = perturbation_levels[i : i + batch_size]

        if hasattr(model, "set_perturbation_levels"):
            cast(Any, model).set_perturbation_levels(batch_levels)
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
        else:
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)

        target_probs = probs[:, target_class].cpu().tolist()
        probabilities.extend(target_probs)
        all_probs.append(probs.cpu().numpy())

    # AUC score
    auc_score = float(auc(perturbation_levels, probabilities))

    all_probs_np = np.concatenate(all_probs, axis=0)

    return probabilities, auc_score, all_probs_np
