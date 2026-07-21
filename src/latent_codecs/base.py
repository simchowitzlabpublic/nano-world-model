from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch


LatentCodecKind = Literal["sd_vae", "webdino", "vjepa2_1", "pixel"]


@dataclass(frozen=True)
class LatentShape:
    channels: int
    height: int
    width: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)


@dataclass(frozen=True)
class LatentCodecConfig:
    kind: LatentCodecKind
    model_path: str | None
    latent_shape: LatentShape
    precision: str
    has_decoder: bool
    input_size: int | None = None
    patch_size: int | None = None
    repo_path: str | None = None
    checkpoint_key: str | None = None


@runtime_checkable
class LatentCodec(Protocol):
    kind: LatentCodecKind
    latent_shape: LatentShape
    has_decoder: bool

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode [B,C,H,W] frames in [-1,1] to [B,Cz,Hz,Wz] latents."""
        ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode [B,Cz,Hz,Wz] latents to [B,C,H,W] frames in [-1,1]."""
        ...

    def requires_grad_(self, requires_grad: bool) -> "LatentCodec":
        """Set trainability for the underlying encoder/decoder modules."""
        ...

    def eval(self) -> "LatentCodec":
        """Put the underlying encoder/decoder modules in eval mode."""
        ...

    def to(self, device: torch.device | str) -> "LatentCodec":
        """Move the underlying encoder/decoder modules to a device."""
        ...
