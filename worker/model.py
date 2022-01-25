import torch.nn as nn
from torchvision import models
import torch


class MobileNet(nn.Module):
    def __init__(self, in_channels=1, out_cls=10, pretrained=0):
        super(MobileNet, self).__init__()
        self.pretrained = pretrained
        self.model = models.mobilenet_v2(pretrained=self.pretrained)

        if in_channels == 1:
            self.conv_1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.features[0][0] = self.conv_1

        self.feature_extractor = self.model.features

        num_ftrs = self.model.classifier[1].in_features
        self.classifier = nn.Linear(in_features=num_ftrs, out_features=out_cls)
        self.dropout = nn.Dropout(p=0.3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = self.avg_pool(features)
        B, F = features.shape[:2]
        out = self.classifier(self.dropout(features.view(B, F)))
        return out


if __name__ == '__main__':
    x = torch.ones((1, 1, 128, 128))
    model = MobileNet(in_channels=1, pretrained=True)
    out = model(x)
