import torch

from .base import LatentShape


class PixelCodec:
    """Identity codec for normalized RGB pixels."""

    kind = "pixel"
    has_decoder = True

    def __init__(self, latent_shape: LatentShape, precision: str):
        self.latent_shape = latent_shape
        self.precision = precision

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return frames

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

    def requires_grad_(self, requires_grad: bool) -> "PixelCodec":
        return self

    def eval(self) -> "PixelCodec":
        return self

    def to(self, device: torch.device | str) -> "PixelCodec":
        return self
