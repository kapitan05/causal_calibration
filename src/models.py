import torch
from torchvision import transforms
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    resnet50,
    resnet101,
    vit_b_16,
    vit_b_32,
)


def load_models(device: torch.device) -> dict[str, torch.nn.Module]:
    """
    Loads ResNet50, ResNet101, ViT-B/16, ViT-B/32 with ImageNet weights.
    Returns dict of models in eval mode.
    """
    print("Loading ResNet50...")
    rn50 = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    rn50.eval()

    print("Loading ResNet101...")
    rn101 = resnet101(weights=ResNet101_Weights.DEFAULT).to(device)
    rn101.eval()

    print("Loading ViT-B/16...")
    vit16 = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)
    vit16.eval()

    print("Loading ViT-B/32...")
    vit32 = vit_b_32(weights=ViT_B_32_Weights.DEFAULT).to(device)
    vit32.eval()

    return {
        "resnet50": rn50,
        "resnet101": rn101,
        "vit_b_16": vit16,
        "vit_b_32": vit32,
    }


def get_preprocessing_transforms() -> transforms.Compose:
    """
    Zwraca standardowe transformacje dla modeli trenowanych na ImageNet.
    Zarówno ResNet50, jak i ViT-B/16 używają tego samego standardu (224x224).
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
