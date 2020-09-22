import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.vgg import make_layers, VGG, cfgs

class VGGLP(VGG):

    def __init__(self, num_classes):
        super(VGG, self).__init__(make_layers(cfgs['D'], batch_norm=True))
        num_features = self.classifier[6].in_features
        features = list(self.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, len(num_classes))])  # Add our layer with 4 outputs
        self.classifier = nn.Sequential(*features)  # Replace the model classifier

