import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder module for VAE."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.training = True

    def forward(self, x):
        """Forward pass of the encoder module."""
        h_ = torch.relu(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        std = torch.exp(0.5 * log_var)
        z = self.reparameterization(mean, std)

        return z, mean, log_var

    def reparameterization(self, mean, std):
        """Reparameterization trick to sample z values."""
        epsilon = torch.rand_like(std)
        z = mean + std * epsilon
        return z


class Decoder(nn.Module):
    """Decoder module for VAE."""

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass of the decoder module."""
        h = torch.relu(self.FC_hidden(x))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class Model(nn.Module):
    """VAE Model."""

    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """Forward pass of the VAE model."""
        z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var
