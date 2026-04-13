import torch
from captum.attr import IntegratedGradients, Occlusion, Saliency, Lime, GradientShap, ShapleyValueSampling, FeatureAblation


class AttributionPipeline:
    """
    A wrapper class for Captum methods that unifies the interface across different
    models.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

        self.methods: dict[str, IntegratedGradients | Saliency | Occlusion | Lime | GradientShap | ShapleyValueSampling | FeatureAblation] = {
            "ig": IntegratedGradients(self.model),
            "saliency": Saliency(self.model),
            "occlusion": Occlusion(self.model),
            "lime": Lime(self.model),
            "gradientshap": GradientShap(self.model),
            "svs": ShapleyValueSampling(self.model),
            "feature_ablation": FeatureAblation(self.model),
        }

    def generate_map(
        self, input_tensor: torch.Tensor, target_class: int, method_name: str
    ) -> torch.Tensor:
        """
        Generates a saliency/attribution map for the given image and target class.
        """
        if method_name not in self.methods:
            raise ValueError(
                f"Nieznana metoda: {method_name}. \
                Wybierz z: {list(self.methods.keys())}"
            )

        method = self.methods[method_name]
        input_tensor.requires_grad_()

        if method_name == "occlusion":
            attributions = method.attribute(
                input_tensor, target=target_class, sliding_window_shapes=(3, 16, 16)
            )
        elif method_name == "ig":
            attributions = method.attribute(
                input_tensor, target=target_class, n_steps=50
            )
        elif method_name == "saliency":
            attributions = method.attribute(input_tensor, target=target_class, abs=True)
        elif method_name == "lime":
            attributions = method.attribute(input_tensor, target=target_class)
        elif method_name == "gradientshap":
            baselines = torch.zeros_like(input_tensor)    
            attributions = method.attribute(
                input_tensor, baselines=baselines, target=target_class, n_samples=50
            )
        elif method_name == "svs":
            attributions = method.attribute(
                input_tensor, target=target_class, n_samples=50
            )
        elif method_name == "feature_ablation":
            attributions = method.attribute(input_tensor, target=target_class)

        if isinstance(attributions, tuple):
            attributions = attributions[0]

        saliency_map = torch.sum(attributions.squeeze(0), dim=0)
        return saliency_map.detach().cpu()
