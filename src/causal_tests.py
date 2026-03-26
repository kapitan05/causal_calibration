import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import auc


def _generate_deletion_sequence(
    image: torch.Tensor,
    saliency_map: torch.Tensor,
    pixels_per_step: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    Generates a sequence of images with progressively removed pixels.
    RISE deletion sequence implementation.
    """
    _, channels, height, width = image.shape
    num_pixels = height * width

    # flatten the inputs
    flat_saliency = saliency_map.flatten()
    flat_image = image.clone().view(channels, -1)  # (3, num_pixels)

    sorted_indices = torch.argsort(flat_saliency, descending=True)

    sequence = [image.clone()]

    # deleteion loop
    for i in range(0, num_pixels, pixels_per_step):
        idx_to_mask = sorted_indices[i : i + pixels_per_step]
        flat_image[:, idx_to_mask] = fill_value
        sequence.append(flat_image.view(1, channels, height, width).clone())

    return torch.cat(sequence, dim=0)


def _generate_insertion_sequence(
    image: torch.Tensor, saliency_map: torch.Tensor, pixels_per_step: int
) -> torch.Tensor:
    """
    Generates a sequence of images progressively restored from a blurred baseline.
    RISE insertion sequence implementation.
    """
    _, channels, height, width = image.shape
    num_pixels = height * width

    blurred_image = TF.gaussian_blur(image, kernel_size=[51, 51], sigma=[50.0, 50.0])

    flat_saliency = saliency_map.flatten()
    flat_original_image = image.view(channels, -1)
    flat_current_image = blurred_image.clone().view(channels, -1)

    sorted_indices = torch.argsort(flat_saliency, descending=True)

    # blerred image as a start of sequence
    sequence = [blurred_image.clone()]

    # insertion loop
    for i in range(0, num_pixels, pixels_per_step):
        idx_to_restore = sorted_indices[i : i + pixels_per_step]
        # restores each channel
        flat_current_image[:, idx_to_restore] = flat_original_image[:, idx_to_restore]
        sequence.append(flat_current_image.view(1, channels, height, width).clone())

    return torch.cat(sequence, dim=0)


@torch.no_grad()
def evaluate_causal_metric(
    model: torch.nn.Module,
    sequence_tensor: torch.Tensor,
    perturbation_levels: list[float],
    target_class: int,
    batch_size: int = 32,
) -> tuple[list[float], float]:
    """
    Executes the Insertion or Deletion algorithm and calculates the AUC score.

    Args:
        model: PyTorch model in evaluation mode.
        image: Input tensor of shape [1, 3, H, W].
        saliency_map: Attribution map of shape [H, W].
        target_class: The index of the class being explained.
        mode: Either "deletion" or "insertion".
        step_fraction: Fraction of total pixels to modify per step (default 0.01 = 1%).
        batch_size: Number of images to process simultaneously for speedup.

    Returns:
        List of probabilities at each step, and the AUC score.
    """

    num_steps = sequence_tensor.size(0)
    probabilities: list[float] = []

    # model evaluation of sequence using batches
    for i in range(0, num_steps, batch_size):
        batch = sequence_tensor[i : i + batch_size]
        batch_levels = perturbation_levels[i : i + batch_size]

        if hasattr(model, "set_perturbation_level"):
            model.set_perturbation_levels(batch_levels)
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
        else:
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)

        target_probs = probs[:, target_class].cpu().tolist()
        probabilities.extend(target_probs)

    # AUC score
    x_axis = np.linspace(0.0, 1.0, len(probabilities)).tolist()
    auc_score = float(auc(x_axis, probabilities))

    return probabilities, auc_score
