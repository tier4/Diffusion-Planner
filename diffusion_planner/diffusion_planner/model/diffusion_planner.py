import torch.nn as nn

from diffusion_planner.model.module.decoder import Decoder
from diffusion_planner.model.module.encoder import Encoder


class Diffusion_Planner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    @property
    def sde(self):
        return self.decoder.sde

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs
