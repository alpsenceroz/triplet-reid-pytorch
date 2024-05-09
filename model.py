from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable

from encoder import ResnetEncoder, VGGEncoder
from decoder import Decoder

## ---------------------- ResNet VAE ---------------------- ##

class VAE(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256, backbone='resnet'):
        super(VAE, self).__init__()

        if backbone == 'resnet':
            self.encoder = ResnetEncoder(fc_hidden1, fc_hidden2, CNN_embed_dim, drop_p)
        elif backbone == 'vgg':
            self.encoder = VGGEncoder(fc_hidden1, fc_hidden2, CNN_embed_dim, drop_p)
        else:
            raise ValueError(f'Backbone not supported: {backbone}')
        
        self.decoder = Decoder(fc_hidden2, CNN_embed_dim)
        

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decoder(z)

        return x_reconst, z, mu, logvar
