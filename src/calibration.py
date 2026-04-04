from asyncio.log import logger
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from scipy import optimize
from torch.utils.data import DataLoader
from tqdm import tqdm

from .attribution import AttributionPipeline
from .generators import SequenceGenerator

"""ReCalX: Core calibration module.
"""


def _cross_entropy_loss(
    temperature: float,
    logits: npt.NDArray[np.float64],
    labels_onehot: npt.NDArray[np.float64],
) -> float:
    """Cross-entropy loss with temperature scaling."""
    n = np.sum(np.exp(logits / temperature), axis=1)
    p = np.clip(np.exp(logits / temperature) / n[:, None], 1e-20, 1 - 1e-20)
    N = p.shape[0]
    return float(-np.sum(labels_onehot * np.log(p)) / N)


def temperature_scaling(
    logits: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    bounds: Tuple[float, float] = (0.05, 5.0),
) -> float:
    """Learn optimal temperature via cross-entropy minimization.

    Args:
        logits: Model logits (n_samples, n_classes)
        labels: True labels (n_samples,)
        bounds: Search bounds for temperature

    Returns:
        Optimal temperature
    """
    n_classes = logits.shape[1]
    labels_onehot = np.eye(n_classes)[labels]

    result = optimize.minimize(
        _cross_entropy_loss,
        x0=1.0,
        args=(logits, labels_onehot),
        method="L-BFGS-B",
        bounds=[bounds],
        options={"ftol": 1e-12},
    )

    return float(result.x[0])


# =============================================================================
# Wrapper dostosowany do obrazów i pętli Causal Tests
# =============================================================================


class ReCalXModel(nn.Module):
    """
    Adaptacja BinnedTemperatureScaling dla testów Insertion/Deletion na obrazach.
    """

    def __init__(self, model: nn.Module, num_bins: int = 10) -> None:
        super().__init__()
        self.model = model
        self.num_bins = num_bins

        # Domyślnie wszystkie temperatury ustawione na 1.0 (brak kalibracji).
        # Zostaną one nadpisane po procesie "uczenia" (nauki temperatur).
        self.temperatures = nn.Parameter(
            torch.ones(num_bins, dtype=torch.float32), requires_grad=False
        )

    def set_perturbation_levels(self, levels: list[float] | torch.Tensor) -> None:
        """
        Ustawia obecny poziom degradacji obrazu (np. 0.3 oznacza usunięcie 30% pikseli).
        W testach Deletion wywołujemy to przed przepuszczeniem batcha przez model.
        """
        if isinstance(levels, list):
            # Przenosimy poziomy na urządzenie modelu (GPU/CPU)
            self.current_levels = torch.tensor(levels, device=self.temperatures.device)
        else:
            self.current_levels = levels.to(self.temperatures.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Przepuszcza dane przez bazowy model i skaluje logity odpowiednim T."""
        logits = self.model(x)

        if not hasattr(self, "current_levels") or self.current_levels is None:
            return logits

        bin_indices = (
            (self.current_levels * self.num_bins).long().clamp(0, self.num_bins - 1)
        )
        temps = self.temperatures[bin_indices].unsqueeze(1)

        return logits / temps

    def load_learned_temperatures(self, learned_temps: list[float]) -> None:
        """Loading temperatures."""
        if len(learned_temps) != self.num_bins:
            raise ValueError(
                f"Expected {self.num_bins} temperatures, got {len(learned_temps)}"
            )

        with torch.no_grad():
            self.temperatures.copy_(torch.tensor(learned_temps, dtype=torch.float32))


@torch.no_grad()
def collect_logits_for_calibration(
    model: torch.nn.Module,
    explainer: AttributionPipeline,
    dataloader: DataLoader,
    sequence_generator: SequenceGenerator,
    device: torch.device,
    num_bins: int = 10,
    **generator_kwargs: any,
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[int]]]:
    """
    Przetwarza zbiór danych, generuje sekwencje degradacji i zbiera logity do koszyków.
    """
    model.eval()
    logits_per_bin: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_bins)}
    labels_per_bin: Dict[int, List[int]] = {i: [] for i in range(num_bins)}

    logger.info("Rozpoczęto zbieranie logitów OOD dla kalibracji...")

    for images, labels in tqdm(dataloader, desc="Przetwarzanie batchy obrazów"):
        images = images.to(device)
        labels = labels.to(device)

        # Iterujemy po obrazach w batchu (generatory pracuja na pojedynczych próbkach)
        # czy sie da batchami, czy trzeba
        for i in range(images.size(0)):
            single_img = images[i : i + 1]
            target_class = labels[i].item()

            saliency_map = explainer.generate_map(single_img, target_class, "ig").to(
                device
            )

            sequence_tensor, exact_levels = sequence_generator(
                single_img, saliency_map, **generator_kwargs
            )

            seq_logits = model(sequence_tensor).cpu().numpy()

            # 4. Przypisywanie logitów do odpowiednich koszyków
            for step_idx, logits in enumerate(seq_logits):
                perturbation_level = exact_levels[step_idx]

                # Zabezpieczenie na wypadek ekstremalnych wartości z inpaintingu/bucketingu
                bin_idx = int(perturbation_level * num_bins)
                bin_idx = min(bin_idx, num_bins - 1)

                logits_per_bin[bin_idx].append(logits)
                labels_per_bin[bin_idx].append(target_class)
    return logits_per_bin, labels_per_bin


def train_and_save_temperatures(
    logits_per_bin: Dict[int, List[np.ndarray]],
    labels_per_bin: Dict[int, List[int]],
    num_bins: int,
    output_path: Path,
) -> List[float]:
    """
    Optymalizuje temperatury dla każdego koszyka i zapisuje wyniki do pliku CSV.
    """
    logger.info("Rozpoczęto optymalizację Temperature Scaling...")
    learned_temperatures: List[float] = []

    for bin_idx in range(num_bins):
        logits_list = logits_per_bin[bin_idx]
        labels_list = labels_per_bin[bin_idx]

        if not logits_list:
            optimal_temp = 1.0
        else:
            np_logits = np.stack(logits_list)
            np_labels = np.array(labels_list)

            # Uczenie ma sens tylko gdy mamy więcej niż jedną próbkę w koszyku
            if len(np_logits) > 1:
                optimal_temp = temperature_scaling(np_logits, np_labels)
            else:
                optimal_temp = 1.0

        learned_temperatures.append(round(optimal_temp, 4))
        bucket_pct = bin_idx * 100 / num_bins
        logger.info(
            f"Bin {bin_idx} (Masked ~{bucket_pct:.1f}%): Optimal T = {optimal_temp:.4f}"
        )

    # Zapis do pliku DataFrame
    df_temps = pd.DataFrame(
        {
            "bin_index": range(num_bins),
            "perturbation_level_start": [i / num_bins for i in range(num_bins)],
            "temperature": learned_temperatures,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_temps.to_csv(output_path, index=False)
    logger.info(f"Temperatures saved to {output_path}")

    return learned_temperatures
