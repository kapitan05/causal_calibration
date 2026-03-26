from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from scipy import optimize

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
