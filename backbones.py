import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import timm

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()


        # encoding components
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.output_size = resnet.fc.in_features
        
        self.resnet = nn.Sequential(*modules)

    def forward(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv
        
        return x
    

class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()

        # encoding components
        vgg = models.vgg16(pretrained=True)
        modules = list(vgg.children())[:-1]  # delete the last fc layer.
        self.vgg = nn.Sequential(*modules)
        self.output_size = vgg.classifier[0].in_features

    def forward(self, x):
        x = self.vgg(x)  # VGG
        x = x.view(x.size(0), -1)  # flatten output of conv
        
        return x
    
    
class DenseNetEncoder(nn.Module):
    def __init__(self):
        super(DenseNetEncoder, self).__init__()

        # encoding components
        densenet = models.densenet121(pretrained=True)
        modules = list(densenet.children())[:-1]  # delete the last fc layer.
        self.densenet = nn.Sequential(*modules)

        # Dummy forward pass to get the output size
        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 128)
            x = self.densenet(x)
        self.output_size = x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.densenet(x)  # DenseNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        return x
    
class SwinEncoder(nn.Module):
    def __init__(self):
        super(SwinEncoder, self).__init__()
        
        # encoding components
        swin_transformer = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.swin_transformer = nn.Sequential(*list(swin_transformer.children())[:-1])  # remove the last layer

        # Dummy forward pass to get the output size
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            x = self.swin_transformer(x)
        self.output_size = x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.swin_transformer(x)  # Swin Transformer
        x = x.view(x.size(0), -1)  # flatten output of conv

        return x
