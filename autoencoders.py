from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from decoder import Decoder

class VAE(nn.Module):
    def __init__(self, input_size=2048, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256, orig_height=256, orig_width=128):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc_hidden2
        self.CNN_embed_dim = CNN_embed_dim
        self.drop_p = drop_p
        self.orig_height = orig_height
        self.orig_width = orig_width
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.fc_hidden1),
            nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.BatchNorm1d(self.fc_hidden2, momentum=0.01),
            nn.ReLU()
        )
        # Latent vectors mu and sigma
        self.fc_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        self.fc_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        
        self.decoder = Decoder(fc_hidden2, CNN_embed_dim)
        

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        

    def forward(self, x):
        x = self.encoder(x)
        if self.drop_p and self.training:
            x = F.dropout(x, p=self.drop_p, training=self.training)

        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        x = self.reparameterize(mu, logvar)
        x_reconst = self.decoder(x, out_size=(self.orig_height, self.orig_width))

        return x_reconst, x, mu, logvar

class AE(nn.Module):
    def __init__(self, input_size=2048, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256, orig_height=256, orig_width=128):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc_hidden2
        self.CNN_embed_dim = CNN_embed_dim
        self.drop_p = drop_p
        self.orig_height = orig_height
        self.orig_width = orig_width
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.fc_hidden1),
            nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.BatchNorm1d(self.fc_hidden2, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.fc_hidden2, self.CNN_embed_dim),
            nn.BatchNorm1d(self.CNN_embed_dim, momentum=0.01),
            nn.ReLU()
        )

        self.decoder = Decoder(CNN_embed_dim, fc_hidden2)
        

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        

    def forward(self, x):
        x = self.encoder(x)
        if self.drop_p and self.training:
            x = F.dropout(x, p=self.drop_p, training=self.training)

        x_reconst = self.decoder(x, out_size=(self.orig_height, self.orig_width))

        return x_reconst, x
