import torch
import torch.nn as nn
from torchvision import models


def get_tb_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Initializes a DenseNet-121 architecture with a frozen feature extraction
    backbone and an optimized classification head for binary image classification.
    """
    # Load pretrained DenseNet-121 model weights
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    # Freeze convolutional feature layers to accelerate training on CPU
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier with custom multi-layer perceptron (MLP)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes)
    )

    return model


if __name__ == "__main__":
    # Internal integrity verification
    test_model = get_tb_model(num_classes=2, pretrained=True)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = test_model(dummy_input)
    print(f"Model integrity verified. Output batch shape: {output.shape}")