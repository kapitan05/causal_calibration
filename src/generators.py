from typing import Any, List, Protocol, Tuple

import torch
import torchvision.transforms.functional as TF


class SequenceGenerator(Protocol):
    """
    Protokół definiujący uniwersalny interfejs dla wszystkich generatorów sekwencji.
    Zapewnia kompatybilność z testami Deletion, Insertion oraz Bucketingiem.
    """

    def __call__(
        self, image: torch.Tensor, saliency_map: torch.Tensor, **kwargs: Any
    ) -> Tuple[torch.Tensor, List[float]]: ...


class TopNDeletionGenerator:
    """Generates standard linear deletion (e.g., top 1%, 2%...)."""

    def __init__(self, step_fraction: float = 0.010, fill_value: float = 0.0):
        self.step_fraction = step_fraction
        self.fill_value = fill_value

    def __call__(
        self, image: torch.Tensor, saliency_map: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float]]:
        height, width = saliency_map.shape[-2:]
        total_pixels = height * width
        pixels_per_step = max(1, int(total_pixels * self.step_fraction))

        sequence = _generate_deletion_sequence(
            image, saliency_map, pixels_per_step, fill_value=self.fill_value
        )

        # Calculate linear exact levels [0.0, 0.1, 0.2...]
        num_steps = sequence.size(0)
        exact_levels = [min(i * self.step_fraction, 1.0) for i in range(num_steps)]

        return sequence, exact_levels


class BucketDeletionGenerator:
    """Generates deletion based on natural significance thresholds (Black pixels)."""

    def __init__(self, num_buckets: int = 10, fill_value: float = 0.0):
        self.num_buckets = num_buckets
        self.fill_value = fill_value

    def __call__(
        self, image: torch.Tensor, saliency_map: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float]]:
        # This calls the function we wrote yesterday
        return _generate_bucket_deletion_sequence(
            image,
            saliency_map,
            num_buckets=self.num_buckets,
            fill_value=self.fill_value,
        )


class TopNInsertionGenerator:
    """Generates standard linear insertion (e.g., revealing top 1%, 2%...)."""

    def __init__(self, step_fraction: float = 0.010):
        self.step_fraction = step_fraction

    def __call__(
        self, image: torch.Tensor, saliency_map: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float]]:
        height, width = saliency_map.shape[-2:]
        total_pixels = (
            height * width
        )  # also counted in _generate_insertion_sequence, but okay
        pixels_per_step = max(1, int(total_pixels * self.step_fraction))

        sequence = _generate_insertion_sequence(image, saliency_map, pixels_per_step)

        # [0.0, step_fraction, 2*step_fraction, ...]
        num_steps = sequence.size(0)
        exact_levels = [min(i * self.step_fraction, 1.0) for i in range(num_steps)]

        return sequence, exact_levels


class BucketInsertionGenerator:
    """Generates insertion based on natural significance thresholds."""

    def __init__(
        self,
        num_buckets: int = 100,
        blur_kernel_size: int = 51,
        blur_sigma: float = 10.0,
    ):
        self.num_buckets = num_buckets
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma

    def __call__(
        self, image: torch.Tensor, saliency_map: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float]]:
        return _generate_bucket_insertion_sequence(
            image=image,
            saliency_map=saliency_map,
            num_buckets=self.num_buckets,
            blur_kernel_size=self.blur_kernel_size,
            blur_sigma=self.blur_sigma,
        )


class BucketInpaintingGenerator:
    """Generates deletion based on natural thresholds, but fills with Inpainting."""

    def __init__(self, num_buckets: int = 10):
        self.num_buckets = num_buckets

    def __call__(
        self, image: torch.Tensor, saliency_map: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float]]:
        # This calls the LaMa inpainting function we wrote yesterday
        return generate_inpainted_bucket_sequence(
            image, saliency_map, num_buckets=self.num_buckets
        )


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


def _generate_bucket_deletion_sequence(
    image: torch.Tensor,
    saliency_map: torch.Tensor,
    num_buckets: int = 10,
    fill_value: float = 0.0,
) -> tuple[torch.Tensor, list[float]]:
    """
    Generates a sequence of images where pixels are removed based on natural
    significance buckets (thresholds) rather than strict percentiles.

    Args:
        image: Original image tensor of shape [1, C, H, W] or [C, H, W].
        saliency_map: Attribution map of shape [H, W].
        num_buckets: Number of discrete significance levels to create.
        fill_value: Value to replace deleted pixels with (0.0 for black).

    Returns:
        A tuple containing:
        - Tensor of shape [num_buckets + 1, C, H, W] with the degraded images.
        - List of floats representing the exact fraction of pixels removed at each step.
    """
    sal_min = saliency_map.min()
    sal_max = saliency_map.max()
    if sal_max > sal_min:
        norm_saliency = (saliency_map - sal_min) / (sal_max - sal_min)
    else:
        norm_saliency = saliency_map

    thresholds = torch.linspace(1.0, 0.0, steps=num_buckets + 1).to(image.device)

    sequence = []
    perturbation_levels = []
    total_pixels = saliency_map.numel()

    img_squeezed = image.squeeze(0)

    for thresh in thresholds:
        delete_mask = norm_saliency >= thresh

        removed_fraction = delete_mask.sum().item() / total_pixels
        perturbation_levels.append(removed_fraction)

        delete_mask_rgb = delete_mask.unsqueeze(0).expand_as(
            img_squeezed
        )  # # [3, 224, 224]

        masked_image = img_squeezed.clone()
        masked_image[delete_mask_rgb] = fill_value

        sequence.append(masked_image)

    return torch.stack(sequence, dim=0), perturbation_levels


def _generate_bucket_insertion_sequence(
    image: torch.Tensor,
    saliency_map: torch.Tensor,
    num_buckets: int = 10,
    blur_kernel_size: int = 51,
    blur_sigma: float = 10.0,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Generates an insertion sequence by gradually revealing original pixels
    over a Gaussian blurred baseline image, based on significance buckets.
    """
    sal_min, sal_max = saliency_map.min(), saliency_map.max()
    if sal_max > sal_min:
        norm_saliency = (saliency_map - sal_min) / (sal_max - sal_min)
    else:
        norm_saliency = saliency_map

    # blurred image
    img_squeezed = image.squeeze(0)
    blurred_image = TF.gaussian_blur(
        img_squeezed, [blur_kernel_size, blur_kernel_size], [blur_sigma, blur_sigma]
    )

    thresholds = torch.linspace(1.0, 0.0, steps=num_buckets + 1).to(image.device)

    sequence = []
    perturbation_levels = []
    total_pixels = saliency_map.numel()

    for thresh in thresholds:
        insert_mask = norm_saliency >= thresh  # [224, 224]

        inserted_fraction = insert_mask.sum().item() / total_pixels
        perturbation_levels.append(inserted_fraction)

        expand_mask = insert_mask.unsqueeze(0).expand_as(img_squeezed)  # [3, 224, 224]

        inserted_image = blurred_image.clone()
        inserted_image[expand_mask] = img_squeezed[expand_mask]

        sequence.append(inserted_image)

    return torch.stack(sequence, dim=0), perturbation_levels


def generate_inpainted_bucket_sequence(
    image: torch.Tensor, saliency_map: torch.Tensor, num_buckets: int = 10
) -> Tuple[torch.Tensor, List[float]]:
    """
    ...
    """
    raise NotImplementedError("This generator is not yet implemented.")
