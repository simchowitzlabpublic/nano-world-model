from pathlib import Path

from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf

from .base import LatentCodecConfig, LatentShape
from .pixel import PixelCodec
from .semantic import VJEPA21LatentCodec, WebDINOLatentCodec
from .sd_vae import SDVAELatentCodec


def _select(cfg, key: str, default=None):
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def get_model_latent_size(cfg) -> int:
    return int(_select(cfg, "model.latent_size", 32))


def get_model_latent_channels(cfg) -> int:
    return int(_select(cfg, "model.latent_channels", 4))


def resolve_latent_codec_config(cfg) -> LatentCodecConfig:
    """Resolve codec settings with legacy config compatibility.

    Older run configs only have top-level `vae_model_path` plus
    `model.latent_size`. Those resolve to the same SD-VAE codec and 4-channel
    latent shape used by the original code.
    """
    if isinstance(cfg, LatentCodecConfig):
        return cfg

    kind = _select(cfg, "latent_codec.kind", None) or _select(cfg, "latent_codec.name", None) or "sd_vae"
    if kind == "vae":
        kind = "sd_vae"
    if kind not in ("sd_vae", "webdino", "vjepa2_1", "pixel"):
        raise NotImplementedError(f"latent_codec.kind={kind!r} is not wired yet")

    model_path = _select(cfg, "latent_codec.model_path", None) or _select(cfg, "latent_codec.vae_model_path", None)
    if model_path is None and kind != "pixel":
        if kind == "sd_vae":
            model_path = _select(cfg, "vae_model_path", "stabilityai/sd-vae-ft-mse")
        else:
            raise ValueError(f"latent_codec.model_path is required for {kind}")
    latent_size = get_model_latent_size(cfg)
    latent_channels = get_model_latent_channels(cfg)

    codec_channels = _select(cfg, "latent_codec.latent_channels", None)
    if codec_channels is not None and int(codec_channels) != latent_channels:
        raise ValueError(
            f"latent_codec.latent_channels={codec_channels} does not match "
            f"model.latent_channels={latent_channels}"
        )

    precision = str(
        _select(cfg, "latent_codec.precision", None)
        or _select(cfg, "experiment.infra.vae_precision", "fp32")
    )
    input_size = _select(cfg, "latent_codec.input_size", None)
    patch_size = _select(cfg, "latent_codec.patch_size", None)

    if kind == "pixel":
        has_decoder = True
        input_size = int(input_size) if input_size is not None else int(_select(cfg, "model.image_size", 256))
        patch_size = None
    elif kind == "sd_vae":
        has_decoder = True
        input_size = int(input_size) if input_size is not None else int(_select(cfg, "model.image_size", 256))
        patch_size = int(patch_size) if patch_size is not None else 8
    elif kind == "webdino":
        has_decoder = False
        patch_size = int(patch_size) if patch_size is not None else 14
        input_size = int(input_size) if input_size is not None else latent_size * patch_size
    else:
        has_decoder = False
        patch_size = int(patch_size) if patch_size is not None else 16
        input_size = int(input_size) if input_size is not None else latent_size * patch_size

    return LatentCodecConfig(
        kind=kind,
        model_path=str(model_path) if model_path is not None else None,
        latent_shape=LatentShape(
            channels=latent_channels,
            height=latent_size,
            width=latent_size,
        ),
        precision=precision,
        has_decoder=has_decoder,
        input_size=input_size,
        patch_size=patch_size,
        repo_path=_select(cfg, "latent_codec.repo_path", None),
        checkpoint_key=_select(cfg, "latent_codec.checkpoint_key", None),
    )


def load_autoencoder_kl(model_path: str) -> AutoencoderKL:
    """Load AutoencoderKL while preserving the old subfolder='vae' behavior."""
    path = Path(str(model_path))
    if path.is_dir() and (path / "config.json").exists():
        return AutoencoderKL.from_pretrained(str(path))
    if path.is_dir() and (path / "vae").is_dir():
        return AutoencoderKL.from_pretrained(str(path), subfolder="vae")

    try:
        return AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    except OSError:
        return AutoencoderKL.from_pretrained(model_path)


def build_sd_vae_codec(vae: AutoencoderKL, cfg) -> SDVAELatentCodec:
    codec_cfg = resolve_latent_codec_config(cfg)
    return SDVAELatentCodec(
        vae=vae,
        precision=codec_cfg.precision,
        latent_shape=codec_cfg.latent_shape,
    )


def build_latent_codec(cfg):
    codec_cfg = resolve_latent_codec_config(cfg)
    if codec_cfg.kind == "pixel":
        return PixelCodec(
            latent_shape=codec_cfg.latent_shape,
            precision=codec_cfg.precision,
        )
    if codec_cfg.kind == "sd_vae":
        vae = load_autoencoder_kl(codec_cfg.model_path)
        return build_sd_vae_codec(vae, cfg)
    if codec_cfg.kind == "webdino":
        return WebDINOLatentCodec(
            model_path=codec_cfg.model_path,
            latent_shape=codec_cfg.latent_shape,
            input_size=codec_cfg.input_size,
            patch_size=codec_cfg.patch_size,
            precision=codec_cfg.precision,
        )
    if codec_cfg.kind == "vjepa2_1":
        return VJEPA21LatentCodec(
            checkpoint_path=codec_cfg.model_path,
            latent_shape=codec_cfg.latent_shape,
            input_size=codec_cfg.input_size,
            patch_size=codec_cfg.patch_size,
            precision=codec_cfg.precision,
            repo_path=codec_cfg.repo_path,
            checkpoint_key=codec_cfg.checkpoint_key or "ema_encoder",
        )
    raise NotImplementedError(f"Unsupported latent codec: {codec_cfg.kind}")
