"""Image classifiers: a small CNN trained from scratch and a VGG16 transfer-learning model."""
import torch.nn as nn
from torchvision import models


def conv_block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                         nn.ReLU(), nn.MaxPool2d(2))


class SmallCNN(nn.Module):
    """4 conv blocks (conv-BN-ReLU-pool) + global average pooling + dropout + linear layer."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64),
                                      conv_block(64, 128), conv_block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def make_vgg16(n_classes=10, pretrained=True):
    """VGG16 with a new classifier head. The convolutional part starts frozen
    (feature extraction); unfreeze `features[24:]` for fine-tuning."""
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.vgg16(weights=weights)
    for p in m.features.parameters():
        p.requires_grad = False
    m.classifier = nn.Sequential(nn.Linear(25088, 512), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(512, n_classes))
    return m
