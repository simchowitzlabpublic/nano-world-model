# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample videos using Diffusion Forcing with context frames from validation set.
Uses first frame(s) as context to predict future frames.
"""
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from tqdm.auto import tqdm

from models import get_models
from diffusion import create_diffusion
from utils.nanowm_utils import find_model
from diffusion.df_sample import dfot_sample, generate_scheduling_matrix
from wm_datasets import create_world_model_dataset
from sampling_utils import encode_frames, decode_latents, save_video, save_comparison_video

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_validation_data(args):
    """Load validation dataset and return dataloader."""
    if not hasattr(args, 'dataset'):
        raise ValueError("Missing required config: dataset")
    if not hasattr(args, 'n_rollout'):
        raise ValueError("Missing required config: n_rollout")

    val_dataset = create_world_model_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        num_frames=args.model.num_frames,
        frame_interval=args.frame_interval,
        image_size=(args.model.image_size, args.model.image_size),
        split='val',
        split_ratio=args.split_ratio,
        normalize_action=args.normalize_action,
        normalize_pixel=True,
        n_rollout=args.n_rollout,
        slice_mode='exhaustive',
        stride=1,
    )

    print(f"Loaded {len(val_dataset)} validation samples")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.experiment.infra.num_workers,
        drop_last=False,
    )

    return val_loader


def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create save directory
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    latent_size = args.model.image_size // 8
    args.model.latent_size = latent_size
    model = get_models(args).to(device)
    
    if args.ckpt:
        state_dict = find_model(args.ckpt)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.ckpt}")
    
    model.eval()
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(device)
    vae.eval()
    vae_precision = getattr(args.experiment.infra, "vae_precision", "fp32")
    
    # Create diffusion
    diffusion = create_diffusion(
        timestep_respacing=str(args.model.num_sampling_steps),
        noise_schedule=args.experiment.diffusion.noise_schedule,
        pred_name=args.experiment.diffusion.pred_name,
        diffusion_steps=args.experiment.diffusion.diffusion_steps,
        snr_gamma=args.experiment.diffusion.snr_gamma,
        zero_terminal_snr=args.experiment.diffusion.zero_terminal_snr,
    )
    
    # Load validation data
    print("Loading validation data...")
    val_loader = load_validation_data(args)
    
    # Sampling parameters
    n_context = args.model.n_context_frames
    num_frames = args.model.num_frames
    
    print(f"\nSampling configuration:")
    print(f"  Context frames: {n_context}")
    print(f"  Total frames: {num_frames}")
    print(f"  Frames to generate: {num_frames - n_context}")
    print(f"  Scheduling mode: {args.model.scheduling_mode}")
    print(f"  Sampling steps: {args.model.num_sampling_steps}")
    
    # Sample from validation set
    sample_count = 0
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing batches")):
        if sample_count >= args.num_samples:
            break
        
        # Debug: print batch keys (only first batch)
        if batch_idx == 0:
            print(f"[DEBUG] Batch keys: {list(batch.keys())}")
            if args.verbose:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"  {k}: {type(v)}")
        
        # Get visual observations: [B, F, C, H, W]
        # batch is a dict when load_raw=False, matching train_pl.py format
        visual = batch['video'].to(device)  # [B, F, C, H, W]
        
        B = visual.shape[0]
        
        # Load action conditioning if enabled
        action = None
        if args.model.use_action:
            action_dim = args.dataset.spec.action_dim
            frame_interval = args.frame_interval
            
            if 'action' in batch:
                # Load action from dataset batch
                action = batch['action'].to(device)  # [B, F, action_dim * frameskip]
                
                if batch_idx == 0:
                    print(f"[Action Loading] Loaded action from dataset, shape: {action.shape}")
                    if args.verbose:
                        print(f"  Action range: min={action.min().item():.3f}, max={action.max().item():.3f}, mean={action.mean().item():.3f}")
            else:
                # Generate random action for testing
                print(f"[Action Loading] WARNING: 'action' not found in batch! Generating random actions instead.")
                if args.use_fp16:
                    action = torch.randn(B, num_frames, action_dim, dtype=torch.float16, device=device)
                else:
                    action = torch.randn(B, num_frames, action_dim, device=device)
                if batch_idx == 0:
                    print(f"[Action Loading] Generated random action, shape: {action.shape}")
        
        # Encode to latent space
        gt_latents = encode_frames(vae, visual, device, vae_precision=vae_precision)
        
        # Debug: print ranges (only first batch)
        if batch_idx == 0 and args.verbose:
            print(f"\n[DEBUG] Data ranges:")
            print(f"  visual (input): min={visual.min().item():.3f}, max={visual.max().item():.3f}")
            print(f"  gt_latents: min={gt_latents.min().item():.3f}, max={gt_latents.max().item():.3f}, std={gt_latents.std().item():.3f}")
            if action is not None:
                print(f"  action: shape={action.shape}, min={action.min().item():.3f}, max={action.max().item():.3f}")
        
        # Extract context frames
        context_latents = gt_latents[:, :n_context]  # [B, n_context, 4, H//8, W//8]
        
        # Generate future frames using DFoT
        shape = (B, num_frames, 4, latent_size, latent_size)
        
        # Prepare model_kwargs with action if enabled
        model_kwargs = dict(y=None, use_fp16=args.use_fp16)
        if action is not None:
            model_kwargs["action"] = action
        
        pred_latents = dfot_sample(
            diffusion=diffusion,
            model=model.forward,
            shape=shape,
            context=context_latents,
            n_context_frames=n_context,
            scheduling_mode=args.model.scheduling_mode,
            num_sampling_steps=args.model.num_sampling_steps,
            model_kwargs=model_kwargs,
            device=device,
            progress=False,
            eta=args.eta,
            clip_denoised=False,
            history_stabilization_level=args.experiment.diffusion.history_stabilization_level,
        )
        
        # Debug: compare latent ranges
        if batch_idx == 0 and args.verbose:
            print(f"  pred_latents: min={pred_latents.min().item():.3f}, max={pred_latents.max().item():.3f}, std={pred_latents.std().item():.3f}")
            # Compare context frames (should match)
            ctx_diff = (pred_latents[:, :n_context] - gt_latents[:, :n_context]).abs().mean().item()
            print(f"  context latent diff (should be ~0): {ctx_diff:.6f}")
        
        # Decode latents
        gt_frames = decode_latents(vae, gt_latents, vae_precision=vae_precision)
        pred_frames = decode_latents(vae, pred_latents, vae_precision=vae_precision)
        
        # Debug: print decoded ranges
        if batch_idx == 0 and args.verbose:
            print(f"  gt_frames (decoded): min={gt_frames.min().item():.3f}, max={gt_frames.max().item():.3f}")
            print(f"  pred_frames (decoded): min={pred_frames.min().item():.3f}, max={pred_frames.max().item():.3f}")
        
        # Save videos
        for i in range(B):
            if sample_count >= args.num_samples:
                break
            
            sample_id = batch_idx * args.batch_size + i
            
            # Save ground truth
            gt_path = save_dir / f"sample_{sample_id:04d}_gt.mp4"
            save_video(gt_frames[i], str(gt_path), fps=args.fps)
            
            # Save prediction
            pred_path = save_dir / f"sample_{sample_id:04d}_pred.mp4"
            save_video(pred_frames[i], str(pred_path), fps=args.fps)
            
            # Save side-by-side comparison
            compare_path = save_dir / f"sample_{sample_id:04d}_compare.mp4"
            save_comparison_video(gt_frames[i], pred_frames[i], str(compare_path), fps=args.fps)
            
            sample_count += 1
            
            if args.verbose:
                print(f"Saved sample {sample_id}")
    
    print(f"\nDone! Saved {sample_count} samples to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diffusion Forcing video prediction sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to config yaml")
    
    # Common overrides (use None as default to detect if user specified)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Directory to save generated videos")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for generation")
    parser.add_argument("--n_context_frames", type=int, default=None,
                        help="Number of context (history) frames")
    parser.add_argument("--scheduling_mode", type=str, default=None,
                        choices=["full_sequence", "pyramid", "sequential"],
                        help="Diffusion Forcing scheduling mode")
    parser.add_argument("--num_sampling_steps", type=int, default=None,
                        help="Number of DDIM sampling steps")
    parser.add_argument("--eta", type=float, default=None,
                        help="DDIM eta (0=deterministic)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # 1. Load base config from file
    conf = OmegaConf.load(args.config)

    # 2. Build CLI overrides mapped to the hierarchical config structure
    cli_args = vars(args)
    cli_overrides = {}

    # Map CLI args to their correct positions in the hierarchy
    model_keys = {'n_context_frames', 'scheduling_mode', 'num_sampling_steps'}
    for key in model_keys:
        if cli_args.get(key) is not None:
            cli_overrides.setdefault('model', {})[key] = cli_args[key]

    # Top-level / non-hierarchical CLI args
    for key in ('ckpt', 'save_path', 'num_samples', 'batch_size', 'eta', 'verbose'):
        val = cli_args.get(key)
        if key == 'verbose':
            if val:
                cli_overrides[key] = val
        elif val is not None:
            cli_overrides[key] = val

    # 3. Merge: config file + CLI overrides
    if cli_overrides:
        conf = OmegaConf.merge(conf, OmegaConf.create(cli_overrides))
    
    # 4. Validate required fields
    if not conf.get('ckpt'):
        raise ValueError(
            "Checkpoint path required. Specify via:\n"
            "  - Config file: ckpt: /path/to/checkpoint\n"
            "  - Command line: --ckpt /path/to/checkpoint"
        )
    
    # 5. Print effective config
    print("=" * 60)
    print("Effective Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(conf))
    print("=" * 60)
    
    main(conf)

