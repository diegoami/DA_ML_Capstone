import torch
import torch.nn as nn

from torchvision.models.vgg import make_layers, VGG

class VGGLP(VGG):
    def __init__(self, num_classes):
        super(VGGLP, self).__init__(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], batch_norm=True))
        num_features = self.classifier[6].in_features
        features = list(self.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, num_classes)])  # Add our layer with 4 outputs
        self.classifier = nn.Sequential(*features)  # Replace the model classifier

