from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class ConvVAE(nn.Module):
    """ Convolutional Variational Autoencoder class
    """
    # input to ConvVAE is resized to 64 * 64 * 3, 
    # each pixel has 3 float values
    # between 0, 1 to represent each of RGB channels
    def __init__(self, N_z=32, batch_size=1):
        super(ConvVAE, self).__init__()

        self.N_z = N_z
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(3, 32, 4, stride = 2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride = 2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride = 2)

        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride = 2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride = 2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride = 2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride = 2)

        self.fc1 = nn.Linear(2 * 2 * 256, self.N_z)
        self.fc2 = nn.Linear(2 * 2 * 256, self.N_z)

        self.fc3 = nn.Linear(self.N_z, 1024)

    def encode(self, x):
        """Encodes the input and returns the encoded vector
        :args x: input vector
        """
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h4 = h4.reshape(-1)
        mu = self.fc1(h4)
        logvar = self.fc2(h4)

        # we take log of var as sigma cannot be negative 
        # while neural network can output negative values
        sigma = torch.exp(logvar * 0.5)

        normal = torch.randn_like(sigma)

        return mu + sigma * normal, mu, logvar

    def decode(self, z):
        """Decodes an encoded vector
        :args z: encoded vector from encode function
        """
        l1 = self.fc3(z)
        l1 = l1.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute(0,3,1,2)
        h1 = F.relu(self.deconv1(l1))
        h2 = F.relu(self.deconv2(h1))
        h3 = F.relu(self.deconv3(h2))
        return torch.sigmoid(self.deconv4(h3))

    def forward(self, x):
        """ Performs a forward pass (i.e., encoding and decoding)
        :args x: input vector
        """
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar


