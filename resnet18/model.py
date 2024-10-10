import torchvision.models as models
from torch import nn
import torch.nn.functional as F

class Net:
    def __init__(self,
                 model_name,
                 num_classes):

        if model_name == 'resnet18':
            print("Using ResNet18")
            self.model = models.resnet18(num_classes=num_classes)

        print('Initializing  model {}'.format(model_name))
