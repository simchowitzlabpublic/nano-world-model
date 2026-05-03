import os
import json
import sys
import math
import random
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins.environments import LightningEnvironment  # noqa: kept for back-compat
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler


def _seed_everything(seed: int) -> None:
    """Seed python/numpy/torch (+ CUDA). Called at the start of training/eval."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN: deterministic algorithms may be slower; prefer reproducibility for ablations.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding so each worker is deterministic given parent seed."""
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

from callbacks import CUDACallback, MetricsLogger
from models import get_models
from wm_datasets import create_train_val_datasets
from diffusion import create_diffusion, sample_training_timesteps
from diffusion.df_sample import dfot_sample
from utils.nanowm_utils import (
    clip_grad_norm_,
    cleanup,
)
from utils.distributed_utils import is_rank_zero
from utils.logger_utils import create_tensorboard_logger, create_wandb_logger
from utils.vae_ops import encode_first_stage, decode_first_stage, vae_autocast_context
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

from .base import BaseExperiment


class NanoWMTrainingModule(LightningModule):
    def __init__(self, args):
        super(NanoWMTrainingModule, self).__init__()
        self.args = args

        self.logger_instance = logging.getLogger(__name__)

        print("[Init] NanoWMTrainingModule: building model", flush=True)
        self.model = get_models(args)

        if args.experiment.pretrained:
            self._load_pretrained_parameters(args)
        self.logger_instance.info(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # torch.compile must wrap *after* any pretrained weight load so that the
        # underlying nn.Module's state_dict layout doesn't shift mid-load. The
        # checkpoint resume path (_load_checkpoint) re-normalizes `_orig_mod.`
        # for cross-mode interop, so toggling this flag between runs is safe.
        if getattr(args.experiment.infra, "compile", False):
            print("[Init] torch.compile(self.model) — first step will pay JIT cost", flush=True)
            self.model = torch.compile(self.model)

        self.diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=args.experiment.diffusion.noise_schedule,
            pred_name=args.experiment.diffusion.pred_name,
            diffusion_steps=args.experiment.diffusion.diffusion_steps,
            snr_gamma=args.experiment.diffusion.snr_gamma,
            zero_terminal_snr=args.experiment.diffusion.zero_terminal_snr,
        )
        print(f"[Init] Loading VAE from: {args.vae_model_path}", flush=True)
        self.vae = AutoencoderKL.from_pretrained(args.vae_model_path)
        # Trust whatever VAE was passed in — read its own scaling factor rather
        # than baking in 0.18215 (SD 1.x) or any other constant. PixArt/SDXL use
        # 0.13025, Flux uses 0.3611, etc.
        self.vae_scale_factor = self.vae.config.scaling_factor
        self._vae_precision = args.experiment.infra.vae_precision
        print(
            f"[Init] VAE loaded, scaling_factor={self.vae_scale_factor}, "
            f"vae_precision={self._vae_precision}",
            flush=True,
        )
        self._sanity_check_vae(args)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.experiment.training.optimizer.lr,
            weight_decay=args.experiment.training.optimizer.weight_decay,
        )
        self.lr_scheduler = None

        self.vae.requires_grad_(False)
        self.model.train()
        print("[Init] NanoWMTrainingModule initialized", flush=True)

    def _sanity_check_vae(self, args):
        """Encode+decode a random batch to catch VAE NaN issues at init time."""
        H = args.model.image_size
        probe = torch.randn(1, 3, H, H)
        with torch.no_grad():
            with vae_autocast_context(self.vae, self._vae_precision):
                z = self.vae.encode(probe).latent_dist.sample()
                recon = self.vae.decode(z).sample
        if not torch.isfinite(z).all():
            raise RuntimeError(
                f"VAE encode produced NaN/Inf at init "
                f"(vae_precision={self._vae_precision}, path={args.vae_model_path}). "
                "If using SDXL/PixArt VAE under bf16, try "
                "experiment.infra.vae_precision=fp32."
            )
        if not torch.isfinite(recon).all():
            raise RuntimeError(
                f"VAE decode produced NaN/Inf at init "
                f"(vae_precision={self._vae_precision}, path={args.vae_model_path})."
            )
        print(f"[Init] VAE sanity: clean under vae_precision={self._vae_precision}", flush=True)

    def _vae_encode(self, x):
        return encode_first_stage(self.vae, x, precision=self._vae_precision)

    def _vae_decode(self, z):
        return decode_first_stage(self.vae, z, precision=self._vae_precision)

    def _load_pretrained_parameters(self, args):
        """Strict-load pretrained weights into self.model.

        Accepts three on-disk formats:
          1. PyTorch-Lightning checkpoint: {"state_dict": {"model.<layer>": ...}}
             (optionally wrapped again by torch.compile as "model._orig_mod.<layer>")
          2. Bare torch state_dict: {"<layer>": ...}
          3. Safetensors file (.safetensors): {"<layer>": ...}

        After prefix stripping the keys and shapes must match self.model exactly —
        this is strict-load so any mismatch is an error the caller must resolve
        (for sim→real we handle action-dim mismatch upstream via pad_action_dim).
        """
        pretrained_path = args.experiment.pretrained
        self.logger_instance.info(f"Loading pretrained weights from {pretrained_path}")

        if str(pretrained_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            raw_sd = load_file(str(pretrained_path))
        else:
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            raw_sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        # PL ckpts pack the whole training module (model + vae + …); filter to
        # `model.*` keys when that prefix is present so we don't spill VAE
        # weights into the transformer's state_dict.
        has_model_prefix = any(k.startswith("model.") for k in raw_sd)
        clean: dict[str, torch.Tensor] = {}
        for k, v in raw_sd.items():
            if has_model_prefix:
                if not k.startswith("model."):
                    continue
                k = k[len("model."):]
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod."):]
            clean[k] = v

        missing, unexpected = self.model.load_state_dict(clean, strict=True)
        assert not missing and not unexpected, (
            f"Pretrained load failed — missing: {missing}, unexpected: {unexpected}"
        )
        self.logger_instance.info(
            f"Loaded {len(clean)} pretrained tensors from {pretrained_path} "
            f"(strict: 0 missing / 0 unexpected)"
        )

    def training_step(self, batch, batch_idx):
        x = batch["video"].to(self.device)
        video_name = batch["video_name"]

        action = None
        if self.args.model.use_action:
            action = batch["action"].to(self.device)

        with torch.no_grad():
            b, _, _, _, _ = x.shape
            x = rearrange(x, "b f c h w -> (b f) c h w").contiguous()
            x = self._vae_encode(x)
            x = rearrange(x, "(b f) c h w -> b f c h w", b=b).contiguous()

        if self.args.model.extras == 78:
            raise ValueError("T2V training is not supported at this moment!")
        elif self.args.model.extras == 2:
            model_kwargs = dict(y=video_name)
        else:
            model_kwargs = dict(y=None)

        if self.args.model.use_action:
            model_kwargs["action"] = action

        diffusion_mode = self.args.experiment.diffusion.mode
        if diffusion_mode == "diffusion_forcing":
            t_shape = (x.shape[0], x.shape[1])
        elif diffusion_mode == "full_seq_diffusion":
            t_shape = (x.shape[0],)
        else:
            raise ValueError(f"Unknown diffusion_mode: {diffusion_mode}. Must be 'full_seq_diffusion' or 'diffusion_forcing'")

        t = sample_training_timesteps(
            t_shape,
            self.diffusion.num_timesteps,
            strategy=self.args.experiment.diffusion.timestep_sampling,
            logit_normal_mean=self.args.experiment.diffusion.logit_normal_mean,
            logit_normal_std=self.args.experiment.diffusion.logit_normal_std,
            device=self.device,
        )

        loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        if self.global_step < self.args.experiment.training.gradient_clip_start_step:
            gradient_norm = clip_grad_norm_(self.model.parameters(), self.args.experiment.training.gradient_clip_norm, clip_grad=False)
        else:
            gradient_norm = clip_grad_norm_(self.model.parameters(), self.args.experiment.training.gradient_clip_norm, clip_grad=True)

        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True, batch_size=x.shape[0])
        self.log("gradient_norm", gradient_norm, logger=True, sync_dist=True, batch_size=x.shape[0])

        if (self.global_step + 1) % self.args.experiment.training.log_every == 0:
            self.logger_instance.info(
                f"(step={self.global_step+1:07d}/epoch={self.current_epoch:04d}) Train Loss: {loss:.4f}, Gradient Norm: {gradient_norm:.4f}"
            )
            for handler in self.logger_instance.handlers:
                handler.flush()
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["video"].to(self.device)
        video_name = batch["video_name"]

        action = None
        if self.args.model.use_action:
            action = batch["action"].to(self.device)

        with torch.no_grad():
            b, _, _, _, _ = x.shape
            x = rearrange(x, "b f c h w -> (b f) c h w").contiguous()
            x = self._vae_encode(x)
            x = rearrange(x, "(b f) c h w -> b f c h w", b=b).contiguous()

            if self.args.model.extras == 2:
                model_kwargs = dict(y=video_name)
            else:
                model_kwargs = dict(y=None)

            if self.args.model.use_action:
                model_kwargs["action"] = action

            diffusion_mode = self.args.experiment.diffusion.mode
            t_shape = (x.shape[0], x.shape[1]) if diffusion_mode == "diffusion_forcing" else (x.shape[0],)
            t = sample_training_timesteps(
                t_shape,
                self.diffusion.num_timesteps,
                strategy=self.args.experiment.diffusion.timestep_sampling,
                logit_normal_mean=self.args.experiment.diffusion.logit_normal_mean,
                logit_normal_std=self.args.experiment.diffusion.logit_normal_std,
                device=self.device,
            )

            loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=x.shape[0])

        return loss

    @torch.no_grad()
    def log_images(self, batch, split="train", sampled_img_num=None, **kwargs):
        x = batch["video"].to(self.device)
        B, F, C, H, W = x.shape

        # No default value - require explicit parameter
        if "cond_frame" not in kwargs:
            n_context = self.args.model.n_context_frames
        else:
            n_context = kwargs["cond_frame"]

        with torch.no_grad():
            x_flat = rearrange(x, "b f c h w -> (b f) c h w").contiguous()
            z = self._vae_encode(x_flat)
            z = rearrange(z, "(b f) c h w -> b f c h w", b=B).contiguous()

        model_kwargs = {}
        if self.args.model.extras == 2:
            model_kwargs["y"] = batch["video_name"].to(self.device) if isinstance(batch["video_name"], torch.Tensor) else batch["video_name"]
        else:
            model_kwargs["y"] = None

        if self.args.model.use_action:
            model_kwargs["action"] = batch["action"].to(self.device)

        # No default values - require explicit parameters
        if "scheduling_mode" not in kwargs:
            scheduling_mode = self.args.model.scheduling_mode
        else:
            scheduling_mode = kwargs["scheduling_mode"]

        if "ddim_steps" not in kwargs:
            num_sampling_steps = self.args.model.num_sampling_steps
        else:
            num_sampling_steps = kwargs["ddim_steps"]

        eval_model = self.model

        z_sample = dfot_sample(
            diffusion=self.diffusion,
            model=eval_model,
            shape=z.shape,
            context=z[:, :n_context],
            n_context_frames=n_context,
            scheduling_mode=scheduling_mode,
            num_sampling_steps=num_sampling_steps,
            model_kwargs=model_kwargs,
            device=self.device,
            progress=False,
            history_stabilization_level=self.args.experiment.diffusion.history_stabilization_level,
        )

        with torch.no_grad():
            z_sample_flat = rearrange(z_sample, "b f c h w -> (b f) c h w").contiguous()
            x_sample = self._vae_decode(z_sample_flat)
            x_sample = rearrange(x_sample, "(b f) c h w -> b c f h w", b=B).contiguous()

            x_gt = rearrange(x, "b f c h w -> b c f h w").contiguous()
            z_flat = rearrange(z, "b f c h w -> (b f) c h w").contiguous()
            x_reconst = self._vae_decode(z_flat)
            x_reconst = rearrange(x_reconst, "(b f) c h w -> b c f h w", b=B).contiguous()
        return {
            "samples": x_sample,
            "reconst": x_reconst,
            "gt": x_gt,
        }

    def on_train_batch_end(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        # PyTorch Lightning steps the scheduler once per optimizer step (not per batch),
        # so warmup_steps / max_steps are optimizer steps directly — no GA multiplier.
        self.lr_scheduler = get_scheduler(
            name="constant",
            optimizer=self.opt,
            num_warmup_steps=self.args.experiment.training.optimizer.lr_warmup_steps,
            num_training_steps=self.args.experiment.training.max_steps,
        )
        return [self.opt], [self.lr_scheduler]

    def on_train_start(self):
        rank = self.global_rank
        experiment_dir = os.getcwd()

        rank_logger = logging.getLogger(f"rank_{rank}")
        rank_logger.setLevel(logging.INFO)
        rank_logger.propagate = False

        log_file = os.path.join(experiment_dir, f"rank_{rank}.log")
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("[%(asctime)s][RANK %(name)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        rank_logger.addHandler(fh)

        if rank == 0:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter("%(message)s"))
            rank_logger.addHandler(ch)

        self.logger_instance = rank_logger
        self.logger_instance.info(f"***** Rank {rank} logging initialized at {log_file} *****")


class TrainExperiment(BaseExperiment):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Use module-level logger (configured by Hydra)
        self.logger = logging.getLogger(__name__)

    def _create_experiment_directory(self):
        """Create experiment directory structure (simplified version)."""
        experiment_dir = HydraConfig.get().runtime.output_dir
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "across_timesteps"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "latest"), exist_ok=True)
        if self.cfg.logger is not None and self.cfg.logger.name == "tensorboard":
            os.makedirs(os.path.join(experiment_dir, "tb"), exist_ok=True)

        if is_rank_zero:
            OmegaConf.save(self.cfg, os.path.join(experiment_dir, "config.yaml"))

        return experiment_dir, checkpoint_dir

    def _create_loggers(self, experiment_dir):
        """Create PyTorch Lightning loggers."""
        loggers = []

        if self.cfg.logger.name == "tensorboard":
            tb_logger = create_tensorboard_logger(
                experiment_dir=experiment_dir,
                name=self.cfg.logger.logger_name
            )
            loggers.append(tb_logger)

        # Optionally create WandB logger
        if self.cfg.wandb.enabled:
            wandb_logger = create_wandb_logger(
                project=self.cfg.wandb.project,
                name=f"{self.cfg.model.name}-F{self.cfg.model.num_frames}-{self.cfg.dataset.name}-{self.cfg.experiment.diffusion.pred_name}",
                experiment_dir=experiment_dir,
                entity=self.cfg.wandb.entity,
                mode=self.cfg.wandb.mode
            )
            if wandb_logger is not None:
                loggers.append(wandb_logger)

        # Log full resolved config as hyperparameters so ablation knobs
        # (model.action_injection.type, model.causal, experiment.diffusion.*, ...)
        # show up in the experiment tracker UI.
        if is_rank_zero and len(loggers) > 0:
            flat_cfg = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=False)
            for logger in loggers:
                try:
                    logger.log_hyperparams(flat_cfg)
                except Exception as exc:
                    # Don't crash training if a logger rejects the payload.
                    logging.getLogger(__name__).warning(
                        f"log_hyperparams failed on {type(logger).__name__}: {exc}"
                    )

        return loggers

    def _apply_fixed_validation_subset(self, val_dataset, experiment_dir):
        loader_cfg = self.cfg.dataset.loader
        subset_path = loader_cfg.validation_fixed_subset_path
        subset_size = loader_cfg.validation_fixed_subset_size
        subset_seed = loader_cfg.validation_fixed_subset_seed

        if subset_path is None and subset_size is None:
            return val_dataset

        if val_dataset.slice_mode != "exhaustive":
            raise ValueError("Fixed validation subset requires slice_mode='exhaustive'")

        if subset_path is None:
            subset_path = os.path.join(experiment_dir, "validation_subset.json")
        elif not os.path.isabs(subset_path):
            subset_path = os.path.join(get_original_cwd(), subset_path)

        if os.path.exists(subset_path):
            with open(subset_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            slice_specs = payload.get("slices", [])
            if len(slice_specs) == 0:
                raise ValueError(f"Empty validation subset file: {subset_path}")
            if subset_size is not None and len(slice_specs) != subset_size:
                raise ValueError(
                    f"Fixed subset size mismatch: requested {subset_size}, "
                    f"but file has {len(slice_specs)} slices ({subset_path})"
                )
            val_dataset.set_fixed_slices(slice_specs)
            if is_rank_zero:
                self.logger.info(f"Loaded fixed validation subset: {len(val_dataset)} slices from {subset_path}")
            return val_dataset

        if subset_size is None:
            raise ValueError("validation_fixed_subset_size is required to generate a new fixed subset")
        if subset_seed is None:
            raise ValueError("validation_fixed_subset_seed is required to generate a new fixed subset")

        slice_specs = val_dataset.sample_fixed_slice_specs(subset_size, seed=subset_seed)
        payload = {
            "dataset": {
                "name": self.cfg.dataset.name,
                "num_frames": self.cfg.model.num_frames,
                "frame_interval": self.cfg.dataset.frame_interval,
                "split_ratio": loader_cfg.split_ratio,
                "val_slice_mode": loader_cfg.val_slice_mode,
                "stride": loader_cfg.stride,
                "random_seed": loader_cfg.random_seed,
            },
            "slices": slice_specs,
        }
        os.makedirs(os.path.dirname(subset_path), exist_ok=True)
        with open(subset_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        val_dataset.set_fixed_slices(slice_specs)
        if is_rank_zero:
            self.logger.info(f"Saved fixed validation subset: {len(val_dataset)} slices to {subset_path}")
        return val_dataset

    def _load_checkpoint(self, args, pl_module):
        """Load checkpoint (simplified version, uses self.logger)."""
        if not args.experiment.resume_from_checkpoint:
            return

        self.logger.info(f"Attempting to load checkpoint from: {args.experiment.resume_from_checkpoint}")

        if not os.path.exists(args.experiment.resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.experiment.resume_from_checkpoint}")

        checkpoint = torch.load(args.experiment.resume_from_checkpoint, map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        has_model_prefix = any(k.startswith("model.") for k in state_dict.keys())
        if not has_model_prefix:
            self.logger.info("No 'model.' prefix detected. Automatically adding prefixes.")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[f"model.{k}"] = v
            state_dict = new_state_dict

        # torch.compile wraps the model in OptimizedModule, whose state_dict
        # prefixes every key with `_orig_mod.`. Normalize the incoming dict so
        # that compile-on / compile-off resumes are interchangeable: strip the
        # prefix when the current model is eager, add it when the current model
        # is compiled. (We only have to look at `model.` keys; vae and others
        # are not compiled.)
        ckpt_has_orig_mod = any("._orig_mod." in k for k in state_dict.keys())
        model_compiled = hasattr(pl_module.model, "_orig_mod")
        if ckpt_has_orig_mod and not model_compiled:
            self.logger.info("Stripping '_orig_mod.' from compiled-saved checkpoint to load into eager model.")
            state_dict = {
                k.replace("model._orig_mod.", "model.", 1) if k.startswith("model._orig_mod.") else k: v
                for k, v in state_dict.items()
            }
        elif model_compiled and not ckpt_has_orig_mod:
            self.logger.info("Inserting '_orig_mod.' into eager-saved checkpoint to load into compiled model.")
            state_dict = {
                k.replace("model.", "model._orig_mod.", 1) if k.startswith("model.") and not k.startswith("model._orig_mod.") else k: v
                for k, v in state_dict.items()
            }

        missing_keys, unexpected_keys = pl_module.load_state_dict(state_dict, strict=False)

        model_missing = [k for k in missing_keys if k.startswith("model.")]
        if len(model_missing) > 0:
            self.logger.error(f"CRITICAL ERROR: Missing model keys in checkpoint: {model_missing}")
            raise RuntimeError(f"Checkpoint loading failed: {len(model_missing)} keys missing.")

        if len(unexpected_keys) > 0:
            self.logger.error(f"CRITICAL ERROR: Unexpected keys in checkpoint: {unexpected_keys}")
            raise RuntimeError(f"Checkpoint loading failed: {len(unexpected_keys)} unexpected keys found.")

        self.logger.info("Successfully loaded model weights with 100% match (VAE excluded).")

    def _setup_common(self, need_train: bool):
        """Shared setup for both training() and evaluate() tasks.

        Returns:
            dict with keys: experiment_dir, checkpoint_dir, loggers,
            train_dataset (None if need_train=False), val_dataset, pl_module,
            eval_callbacks (pre-built list with CUDACallback + optional
            MetricsLogger + LearningRateMonitor, but no checkpoint callbacks).
        """
        args = self.cfg
        seed = args.experiment.infra.seed
        _seed_everything(seed)
        self._seed = seed  # for dataloader generator

        experiment_dir, checkpoint_dir = self._create_experiment_directory()
        loggers = self._create_loggers(experiment_dir)

        if not hasattr(args, "dataset"):
            raise ValueError("Missing dataset in config. Define configs/dataset/...")

        dataset_cfg = OmegaConf.to_container(args.dataset, resolve=True)
        if "name" not in dataset_cfg:
            raise ValueError("Missing required config: dataset.name")
        if "loader" not in dataset_cfg:
            raise ValueError("Missing required config: dataset.loader")

        dataset_name = dataset_cfg["name"]
        loader_cfg = dataset_cfg["loader"]

        # Fast path for eval-only: if fixed subset JSON exists, skip full
        # slice indexing and build val dataset directly from cached specs.
        _used_fast_path = False
        if not need_train:
            subset_path = loader_cfg.get("validation_fixed_subset_path")
            if subset_path and os.path.exists(subset_path):
                with open(subset_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                slice_specs = payload.get("slices", [])
                if slice_specs:
                    from wm_datasets.world_model_dataset import create_eval_only_dataset
                    val_dataset = create_eval_only_dataset(
                        dataset_name=dataset_name,
                        num_frames=args.model.num_frames,
                        frame_interval=args.dataset.frame_interval,
                        image_size=args.model.image_size,
                        precomputed_slices=slice_specs,
                        **loader_cfg,
                    )
                    train_dataset = None
                    _used_fast_path = True
                    if is_rank_zero:
                        print(f"[Data] Fast path: loaded {len(slice_specs)} precomputed slices, skipped full indexing", flush=True)

        if not _used_fast_path:
            train_dataset, val_dataset = create_train_val_datasets(
                dataset_name=dataset_name,
                num_frames=args.model.num_frames,
                frame_interval=args.dataset.frame_interval,
                image_size=args.model.image_size,
                **loader_cfg,
            )
            if is_rank_zero:
                print("[Data] Datasets created", flush=True)

            val_dataset = self._apply_fixed_validation_subset(val_dataset, experiment_dir)
        if (
            args.dataset.loader.validation_size is not None
            and args.dataset.loader.validation_fixed_subset_path is None
            and args.dataset.loader.validation_fixed_subset_size is None
        ):
            indices = list(range(min(len(val_dataset), args.dataset.loader.validation_size)))
            val_dataset = Subset(val_dataset, indices)
            if is_rank_zero:
                self.logger.info(f"Using {len(val_dataset)} validation samples")

        pl_module = NanoWMTrainingModule(args)
        if args.experiment.resume_from_checkpoint:
            self._load_checkpoint(args, pl_module)

        callbacks_list = [CUDACallback()]
        # LearningRateMonitor requires a logger to write to; skip it when no
        # logger is configured (wandb.enabled=false and logger.name != "tensorboard").
        if is_rank_zero and len(loggers) > 0:
            callbacks_list.append(LearningRateMonitor())

        eval_cfg = args.experiment.evaluation
        eval_metrics_cfg = eval_cfg.metrics
        if eval_metrics_cfg.evaluate:
            i3d_path = eval_metrics_cfg.i3d_model_path
            if i3d_path and not os.path.isabs(i3d_path):
                i3d_path = os.path.join(get_original_cwd(), i3d_path)
            if is_rank_zero:
                print(
                    f"[Eval] Initializing LPIPS/FID/FVD evaluator "
                    f"(i3d_model_path={i3d_path})",
                    flush=True,
                )
            log_images_kwargs = {}
            eval_scheduling_mode = eval_cfg.get("scheduling_mode")
            if eval_scheduling_mode is not None:
                log_images_kwargs["scheduling_mode"] = eval_scheduling_mode

            callbacks_list.append(MetricsLogger(
                env=args.dataset,
                log_every_n_train_steps=eval_metrics_cfg.log_every_n_train_steps,
                buffer_size=eval_metrics_cfg.buffer_size,
                i3d_model_path=i3d_path,
                max_batchsize=eval_metrics_cfg.max_batchsize,
                log_images_kwargs=log_images_kwargs,
                save_dir=os.path.join(experiment_dir, "eval_videos") if eval_cfg.save_videos else None,
                evaluate=eval_metrics_cfg.evaluate,
            ))

        return {
            "experiment_dir": experiment_dir,
            "checkpoint_dir": checkpoint_dir,
            "loggers": loggers,
            "train_dataset": train_dataset if need_train else None,
            "val_dataset": val_dataset,
            "pl_module": pl_module,
            "eval_callbacks": callbacks_list,
        }

    def _make_val_loader(self):
        args = self.cfg
        return dict(
            batch_size=args.experiment.training.batch_size,
            shuffle=False,
            num_workers=args.experiment.infra.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=_seed_worker if args.experiment.infra.num_workers > 0 else None,
        )

    def training(self):
        """Main training task."""
        args = self.cfg
        common = self._setup_common(need_train=True)
        experiment_dir = common["experiment_dir"]
        checkpoint_dir = common["checkpoint_dir"]
        loggers = common["loggers"]
        train_dataset = common["train_dataset"]
        val_dataset = common["val_dataset"]
        pl_module = common["pl_module"]
        callbacks_list = common["eval_callbacks"]

        loader_seed_gen = torch.Generator()
        loader_seed_gen.manual_seed(self._seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.experiment.training.batch_size,
            shuffle=True,
            num_workers=args.experiment.infra.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.experiment.infra.num_workers > 0,
            prefetch_factor=4 if args.experiment.infra.num_workers > 0 else None,
            worker_init_fn=_seed_worker if args.experiment.infra.num_workers > 0 else None,
            generator=loader_seed_gen,
        )
        val_loader = DataLoader(
            val_dataset,
            persistent_workers=args.experiment.infra.num_workers > 0,
            prefetch_factor=4 if args.experiment.infra.num_workers > 0 else None,
            **self._make_val_loader(),
        )
        if is_rank_zero:
            print("[Data] DataLoaders created", flush=True)
            self.logger.info(f"Training dataset contains {len(train_dataset)} videos")
            self.logger.info(f"Validation dataset contains {len(val_dataset)} videos")
            self.logger.info(f"One epoch iteration {math.ceil(len(train_loader))} steps")

        ckpt_cfg = args.experiment.training.checkpointing
        across_timesteps_cfg = ckpt_cfg.across_timesteps
        across_timesteps_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, "across_timesteps"),
            filename=across_timesteps_cfg.filename,
            save_top_k=across_timesteps_cfg.save_top_k,
            every_n_train_steps=across_timesteps_cfg.every_n_train_steps,
            save_on_train_epoch_end=across_timesteps_cfg.save_on_train_epoch_end,
            save_weights_only=across_timesteps_cfg.save_weights_only,
        )
        callbacks_list.append(across_timesteps_checkpoint)

        latest_cfg = ckpt_cfg.latest
        latest_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, "latest"),
            filename=latest_cfg.filename,
            save_top_k=latest_cfg.save_top_k,
            every_n_train_steps=latest_cfg.every_n_train_steps,
            save_on_train_epoch_end=latest_cfg.save_on_train_epoch_end,
            save_weights_only=latest_cfg.save_weights_only,
        )
        callbacks_list.append(latest_checkpoint)

        num_gpus = torch.cuda.device_count()
        num_nodes = args.experiment.infra.get("num_nodes", 1)
        val_check_interval = args.experiment.training.val_every_n_steps

        trainer = Trainer(
            accelerator="gpu",
            devices=num_gpus,
            strategy=("ddp_find_unused_parameters_false" if num_gpus > 1 else "auto"),
            num_nodes=num_nodes,
            enable_checkpointing=True,
            max_steps=args.experiment.training.max_steps,
            logger=loggers,
            callbacks=callbacks_list,
            log_every_n_steps=args.experiment.training.log_every,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=None,
            accumulate_grad_batches=args.experiment.training.gradient_accumulation,
            precision="bf16" if args.experiment.infra.mixed_precision else "32",
            num_sanity_val_steps=0,
        )

        # exec() already dispatches on self.tasks, so training() only does fit.
        # Evaluation is handled by the separate evaluate() method.
        trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        pl_module.model.eval()
        self.logger.info("Done!")

        cleanup()

    def evaluate(self):
        """Standalone evaluation task - runs validation only without training."""
        args = self.cfg

        if not args.experiment.resume_from_checkpoint:
            raise ValueError(
                "resume_from_checkpoint is required for evaluation task. "
                "Please specify a checkpoint path."
            )

        common = self._setup_common(need_train=False)
        loggers = common["loggers"]
        val_dataset = common["val_dataset"]
        pl_module = common["pl_module"]
        callbacks_list = common["eval_callbacks"]

        val_loader = DataLoader(val_dataset, **self._make_val_loader())

        num_gpus = torch.cuda.device_count()
        num_nodes = args.experiment.infra.get("num_nodes", 1)
        trainer = Trainer(
            accelerator="gpu",
            devices=num_gpus,
            strategy=("ddp_find_unused_parameters_false" if num_gpus > 1 else "auto"),
            num_nodes=num_nodes,
            enable_checkpointing=False,  # No checkpointing during evaluation
            logger=loggers,
            callbacks=callbacks_list,
            precision="bf16" if args.experiment.infra.mixed_precision else "32",
            num_sanity_val_steps=0,
        )

        self.logger.info("***** Running Evaluation *****")
        trainer.validate(pl_module, dataloaders=val_loader)

        pl_module.model.eval()
        self.logger.info("Evaluation Done!")

        cleanup()
