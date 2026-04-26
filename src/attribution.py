import torch
from captum.attr import IntegratedGradients, Occlusion, Saliency, Lime, GradientShap, ShapleyValueSampling, FeatureAblation
from rise import RISE  # your RISE class

class AttributionPipeline:
    def __init__(self, model: torch.nn.Module, input_size=(224, 224)) -> None:
        self.model = model

        # RISE setup
        self._rise = RISE(model, input_size)
        self._rise.generate_masks(N=5000, s=10, p1=0.1)

        self.methods: dict = {
            "ig": IntegratedGradients(self.model),
            "saliency": Saliency(self.model),
            "occlusion": Occlusion(self.model),
            "lime": Lime(self.model),
            "gradientshap": GradientShap(self.model),
            "svs": ShapleyValueSampling(self.model),
            "feature_ablation": FeatureAblation(self.model),
            "rise": None,  # handled separately
        }

    def _run_rise(self, input_tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        self._rise.eval()
        with torch.no_grad():
            sal = self._rise(input_tensor.cuda())[target_class]  # (H, W)
        return sal.cpu()

    def create_patch_mask(self, input_tensor, patch_size=8):
        _, _, H, W = input_tensor.shape
        mask = torch.zeros((1, H, W), dtype=torch.long, device=input_tensor.device)
        idx = 0
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                mask[:, i:i+patch_size, j:j+patch_size] = idx
                idx += 1
        return mask

    def generate_map(
        self, input_tensor: torch.Tensor, target_class: int, method_name: str
    ) -> torch.Tensor:
        if method_name not in self.methods:
            raise ValueError(
                f"Unknown method: {method_name}. Choose from: {list(self.methods.keys())}"
            )

        # RISE exits early — it has its own interface
        if method_name == "rise":
            return self._run_rise(input_tensor, target_class)

        method = self.methods[method_name]
        input_tensor.requires_grad_()

        if method_name == "occlusion":
            attributions = method.attribute(input_tensor, target=target_class, sliding_window_shapes=(3, 16, 16))
        elif method_name == "ig":
            attributions = method.attribute(input_tensor, target=target_class, n_steps=50)
        elif method_name == "saliency":
            attributions = method.attribute(input_tensor, target=target_class, abs=True)
        elif method_name == "lime":
            attributions = method.attribute(input_tensor, target=target_class)
        elif method_name == "gradientshap":
            baselines = torch.zeros_like(input_tensor)
            attributions = method.attribute(input_tensor, baselines=baselines, target=target_class, n_samples=50)
        elif method_name == "svs":
            mask = self.create_patch_mask(input_tensor)
            attributions = method.attribute(input_tensor, target=target_class, n_samples=50, feature_mask=mask)
        elif method_name == "feature_ablation":
            mask = self.create_patch_mask(input_tensor)
            attributions = method.attribute(input_tensor, target=target_class, feature_mask=mask)

        if isinstance(attributions, tuple):
            attributions = attributions[0]

        saliency_map = torch.sum(attributions.squeeze(0), dim=0)
        return saliency_map.detach().cpu()