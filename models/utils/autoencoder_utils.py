import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dim, dim_mults, shape:tuple, in_channels) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([
                            nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(),
                            nn.BatchNorm2d(8),])
        for multiplier in range(dim_mults):
            self.encoder.append(nn.ModuleList([
                                    nn.Conv2d(8 * multiplier, 16 * multiplier, 3, padding=1), nn.ReLU(),
                                    nn.BatchNorm2d(8*multiplier),
                                    nn.MaxPool2d(2)
                                    ]))
        
        self.encoder.append(nn.ModuleList([
                                Flatten(),
                                nn.Linear(128 * shape[0] * shape[1], 32 * shape[0]*shape[1]),
                                nn.ReLU(),
                                nn.Linear(32*shape[0]*shape[1], latent_dim),
                                nn.ReLU()]))
        self.hidden2mu = nn.Linear(28,28)
        self.hidden2log_var = nn.Linear(28,28)

        
    def reparametrize(self, mu, log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size = (mu.size(0), mu.size(1)))
        z = z.type_as(mu) # Setting z to be .cuda when using GPU training 
        return mu + sigma * z
    

    def encode(self,x):
       hidden = self.encoder(x)
       mu = self.hidden2mu(hidden)
       log_var = self.hidden2log_var(hidden)
       return mu, log_var

    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        return hidden, mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, dim_mults:list, shape, out_channels) -> None:
        super().__init__()
        self.decoder = nn.ModuleList([ 
                            nn.Linear(latent_dim, 32 * shape[0] * shape[1]), 
                            nn.ReLU(),
                            nn.BatchNorm1d(32 * shape * shape),
                            nn.Linear(32 * shape[0] * shape[1], 128 * shape[0] * shape[1]),
                            nn.ReLU(), 
                            nn.BatchNorm1d(128 * shape[0] * shape[1]),
                            Stack(128, 3, 3)])
        dim_mults.reverse()
        for multiplier in dim_mults:
            self.decoder.append(nn.ModuleList(
                                    [nn.ConvTranspose2d(16*multiplier, 8*multiplier, 3, 2), 
                                    nn.ReLU(), 
                                    nn.BatchNorm2d(8*multiplier)]))

        self.decoder.append(nn.ModuleList([
                                nn.Conv2d(8, out_channels, 3, padding=1),
                                nn.Tanh()]))

    def forward(self, latent):
        return self.decoder(latent)


class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)
    

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)