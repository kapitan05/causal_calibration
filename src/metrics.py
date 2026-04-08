from typing import Any

import numpy as np
import numpy.typing as npt


def calculate_ece(
    probs: npt.NDArray[np.float64], labels: npt.NDArray[np.int_], n_bins: int = 10
) -> float:
    """
    Calculates Expected Calibration Error (ECE).

    Args:
        probs: Numpy array [N, num_classes] with probabilities (after softmax).
        labels: Numpy array [N] with true labels.
        n_bins: Number of bins for uncertainty quantification (default is 10).
    """
    # safeguard against empty input
    if len(probs) == 0:
        return 0.0

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0

    for i in range(n_bins):
        in_bin = (confidences >= bins[i]) & (confidences < bins[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece_val += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return float(ece_val)


def calculate_tace(
    probs: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    n_bins: int = 15,
    threshold: float = 0.01,
) -> float:
    """
    Calculates the Thresholded Adaptive Calibration Error (TACE) for Top-1 predictions.

    Args:
        probs: Numpy array of shape [N, num_classes] with softmax probabilities.
        labels: Numpy array of shape [N] with ground truth labels.
        n_bins: Number of adaptive ranges (R). The paper recommends 15.
        threshold: The epsilon value to filter out tiny predictions.

    Returns:
        The TACE score as a float.
    """
    if len(probs) == 0:
        return 0.0

    # Extract Top-1 confidences and predictions
    confidences = np.max(probs, axis=1)  # [N]
    predictions = np.argmax(probs, axis=1)  # [N]
    accuracies = (predictions == labels).astype(np.float64)  # [N]

    # thresholding to filter out low-confidence predictions
    valid_mask = confidences >= threshold
    filtered_confs = confidences[valid_mask]
    filtered_accs = accuracies[valid_mask]

    if len(filtered_confs) == 0:
        return 0.0

    sorted_indices = np.argsort(filtered_confs)
    sorted_confs = filtered_confs[sorted_indices]
    sorted_accs = filtered_accs[sorted_indices]

    confs_split = np.array_split(sorted_confs, n_bins)
    accs_split = np.array_split(sorted_accs, n_bins)

    tace_val = 0.0

    for conf_bin, acc_bin in zip(confs_split, accs_split):
        if len(conf_bin) > 0:
            bin_conf = np.mean(conf_bin)
            bin_acc = np.mean(acc_bin)

            tace_val += np.abs(bin_conf - bin_acc)

    return float(tace_val / n_bins)


def process_calibration_metrics(
    probs_base_dict: dict[int, list[np.ndarray[Any, Any]]],
    probs_cal_dict: dict[int, list[np.ndarray[Any, Any]]],
    labels_dict: dict[int, list[int]],
    num_bins: int,
    needs_calibration: bool,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Calculates calibration metrics (ECE, TACE) for base and calibrated
    models across bins.

    Args:
        probs_base_dict: Dictionary mapping bin index to a list of base model
        probability arrays.
        probs_cal_dict: Dictionary mapping bin index to a list of calibrated model
        probability arrays.
        labels_dict: Dictionary mapping bin index to a list of true labels.
        num_bins: Total number of bins.
        needs_calibration: Flag indicating if the calibrated model should be evaluated.

    Returns:
        A tuple of four lists: (base_ECE, calibrated_ECE, base_TACE, calibrated_TACE).
    """
    ece_base: list[float] = []
    ece_cal: list[float] = []
    tace_base: list[float] = []
    tace_cal: list[float] = []

    for bin_idx in range(num_bins):
        labels_arr = np.array(labels_dict[bin_idx])

        if len(labels_arr) > 0:
            base_probs_arr = np.array(probs_base_dict[bin_idx])
            ece_base.append(calculate_ece(base_probs_arr, labels_arr))
            tace_base.append(calculate_tace(base_probs_arr, labels_arr))

            if needs_calibration:
                cal_probs_arr = np.array(probs_cal_dict[bin_idx])
                ece_cal.append(calculate_ece(cal_probs_arr, labels_arr))
                tace_cal.append(calculate_tace(cal_probs_arr, labels_arr))
            else:
                ece_cal.append(0.0)
                tace_cal.append(0.0)
        else:
            ece_base.append(0.0)
            ece_cal.append(0.0)
            tace_base.append(0.0)
            tace_cal.append(0.0)

    return ece_base, ece_cal, tace_base, tace_cal
