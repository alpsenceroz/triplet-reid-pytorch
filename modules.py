from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable

from encoder import Encoder
from decoder import Decoder

## ---------------------- ResNet VAE ---------------------- ##

class ResNet_VAE(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(ResNet_VAE, self).__init__()

        self.encoder = Encoder(fc_hidden1, fc_hidden2, CNN_embed_dim, drop_p)
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
