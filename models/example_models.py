import torch.nn as nn
from torchvision import models
from .model_wrapper import ModelWrapper

def get_resnet18(num_classes: int = 200, pretrained: bool = True) -> ModelWrapper:
    """
    Returns a wrapped ResNet-18 model.

    Args:
        num_classes (int): Number of output classes. Defaults to 200.
        pretrained (bool): Whether to load ImageNet pretrained weights. Defaults to True.

    Returns:
        ModelWrapper: Wrapped ResNet-18.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return ModelWrapper(model, "resnet18", pretrained)

def get_resnet34(num_classes: int = 200, pretrained: bool = True) -> ModelWrapper:
    """
    Returns a wrapped ResNet-34 model.

    Args:
        num_classes (int): Number of output classes. Defaults to 200.
        pretrained (bool): Whether to load ImageNet pretrained weights. Defaults to True.

    Returns:
        ModelWrapper: Wrapped ResNet-34.
    """
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    model = models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return ModelWrapper(model, "resnet34", pretrained)

def get_mobilenet_v2(num_classes: int = 200, pretrained: bool = True) -> ModelWrapper:
    """
    Returns a wrapped MobileNetV2 model.

    Args:
        num_classes (int): Number of output classes. Defaults to 200.
        pretrained (bool): Whether to load ImageNet pretrained weights. Defaults to True.

    Returns:
        ModelWrapper: Wrapped MobileNetV2.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return ModelWrapper(model, "mobilenet_v2", pretrained)

def get_efficientnet_v2_s(num_classes: int = 200, pretrained: bool = True) -> ModelWrapper:
    """
    Returns a wrapped EfficientNetV2-S model.

    Args:
        num_classes (int): Number of output classes. Defaults to 200.
        pretrained (bool): Whether to load ImageNet pretrained weights. Defaults to True.

    Returns:
        ModelWrapper: Wrapped EfficientNetV2-S.
    """
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return ModelWrapper(model, "efficientnet_v2_s", pretrained)
