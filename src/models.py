import torch
from torchvision import transforms
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16


def load_models(device: torch.device) -> dict[str, torch.nn.Module]:
    """
    Pobiera i konfiguruje modele ResNet50 oraz ViT-B/16.
    Zwraca słownik z gotowymi do pracy modelami w trybie ewaluacji.
    """
    print("Pobieranie wag modelu ResNet50...")
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    resnet.eval()

    print("Pobieranie wag modelu ViT-B/16...")
    vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)
    vit.eval()

    return {
        "resnet50": resnet,
        "vit_b_16": vit,
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
