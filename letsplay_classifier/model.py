import torch
import torch.nn as nn

from torchvision.models.vgg import make_layers, VGG, cfgs

"""
The VGG classifier adapted for our use case
"""
class VGGLP(VGG):
    
    def __init__(self, num_classes, layer_cfg='D'):
        super(VGGLP, self).__init__(make_layers(cfgs[layer_cfg], batch_norm=True))
        num_features = self.classifier[6].in_features
        features = list(self.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, num_classes)])  # Add our layer with 4 outputs
        self.classifier = nn.Sequential(*features)  # Replace the model classifier

