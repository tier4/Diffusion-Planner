import torch.nn as nn

from diffusion_planner.model.module.decoder import Decoder
from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.plantf_decoder import PlanTFDecoder


def build_decoder(config):
    # getattr: args.json from runs older than decoder_type defaults to diffusion
    decoder_type = getattr(config, "decoder_type", "diffusion")
    if decoder_type == "diffusion":
        return Decoder(config)
    if decoder_type == "plantf":
        return PlanTFDecoder(config)
    raise ValueError(f"Unknown decoder type: {decoder_type}")


class Diffusion_Planner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = build_decoder(config)

    @property
    def sde(self):
        return self.decoder.sde

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs
