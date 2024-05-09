from torch import nn
from torch.nn import functional as F
from torchvision import models
import timm

class ResNetEncoder(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=256, drop_p=None):
        super(ResNetEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim
        self.relu = nn.ReLU(inplace=True)
        self.drop_p = drop_p

        # encoding components
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

    def forward(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        if self.drop_p and self.training:
            x = F.dropout(x, p=self.drop_p, training=self.training)
            
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar
    

class VGGEncoder(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=256, drop_p=None):
        super(VGGEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim
        self.relu = nn.ReLU(inplace=True)
        self.drop_p = drop_p

        # encoding components
        vgg = models.vgg16(pretrained=True)
        modules = list(vgg.children())[:-1]  # delete the last fc layer.
        self.vgg = nn.Sequential(*modules)
        self.fc1 = nn.Linear(vgg.classifier[0].in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

    def forward(self, x):
        x = self.vgg(x)  # VGG
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        if self.drop_p and self.training:
            x = F.dropout(x, p=self.drop_p, training=self.training)
            
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar
    
class DenseNetEncoder(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=256, drop_p=None):
        super(DenseNetEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim
        self.relu = nn.ReLU(inplace=True)
        self.drop_p = drop_p

        # encoding components
        densenet = models.densenet121(pretrained=True)
        modules = list(densenet.children())[:-1]  # delete the last fc layer.
        self.densenet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(densenet.classifier.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

    def forward(self, x):
        x = self.densenet(x)  # DenseNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        if self.drop_p and self.training:
            x = F.dropout(x, p=self.drop_p, training=self.training)
            
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar
    
class SwinEncoder(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=256, drop_p=None):
        super(SwinEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim
        self.relu = nn.ReLU(inplace=True)
        self.drop_p = drop_p

        # encoding components
        swin_transformer = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.swin_transformer = nn.Sequential(*list(swin_transformer.children())[:-1])  # remove the last layer
        self.fc1 = nn.Linear(swin_transformer.head.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

    def forward(self, x):
        x = self.swin_transformer(x)  # Swin Transformer
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        if self.drop_p and self.training:
            x = F.dropout(x, p=self.drop_p, training=self.training)
            
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar