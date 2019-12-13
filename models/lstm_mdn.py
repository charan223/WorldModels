import torch
from torch import nn
from torch.nn import functional as F

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")


class LSTM_MDN(nn.Module):
    def __init__(self, seq_size=100, z_dim=32, temperature=1.15, hidden_units=256, n_gaussians=5, hidden_dim=256):
        super(LSTM_MDN, self).__init__()

        self.seq_size = seq_size
        self.z_dim = z_dim
        self.temperature = temperature
        self.hidden_units = hidden_units
        self.n_gaussians = n_gaussians
        self.hidden_dim = hidden_dim

        hidden = torch.zeros(1, seq_size, self.hidden_units, device=DEVICE)
        cell = torch.zeros(1, seq_size, self.hidden_units, device=DEVICE)
        self.hidden = hidden, cell

        self.fc1 = nn.Linear(self.z_dim + 3, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_units, 1)
        self.pi = nn.Linear(self.hidden_units, self.n_gaussians * self.z_dim)
        self.mu = nn.Linear(self.hidden_units, self.n_gaussians * self.z_dim)
        self.sigma = nn.Linear(self.hidden_units, self.n_gaussians * self.z_dim)

    def forward(self, x):
        self.lstm.flatten_parameters()
        seq = x.size()[1]

        # LSTM
        x = F.relu(self.fc1(x))
        z, self.hidden = self.lstm(x, self.hidden)

        # MDN to get the priors, means and covariance of the gaussian mixture
        pi = self.pi(z).view(-1, seq, self.n_gaussians, self.z_dim)
        pi = F.softmax(pi, dim=2)
        pi = pi / self.temperature
        mu = self.mu(z).view(-1, seq, self.n_gaussians, self.z_dim)
        sigma = torch.exp(self.sigma(z)).view(-1, seq, self.n_gaussians, self.z_dim)
        sigma = sigma * (self.temperature ** 0.5)

        return pi, sigma, mu
