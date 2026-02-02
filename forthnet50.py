

import torch
import torch.nn as nn
from torchvision import models


class ForthNet50(nn.Module):


    def __init__(self, num_classes: int = 4, dropout_p: float = 0.3):
        super().__init__()

        # Load ImageNet-pretrained ResNet-50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Keep everything except the final fc
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,  
            backbone.layer2,  
            backbone.layer3,  
            backbone.layer4,  
        )
        self.avgpool = backbone.avgpool

        in_features = backbone.fc.in_features  

        # Lightweight dense head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )

        self._freeze_lower_layers()

    def _freeze_lower_layers(self):

        # Freeze everything
        for param in self.features.parameters():
            param.requires_grad = False

        # Unfreeze conv5_x (layer4)
        for param in self.features[7].parameters():
            param.requires_grad = True

        # Classifier is trainable by default

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
