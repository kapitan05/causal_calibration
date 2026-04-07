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
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

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
