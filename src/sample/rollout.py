# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Long-term rollout using Diffusion Forcing with a sliding window and batching.
Uses the sliced dataset for evaluation consistency.
"""
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import torch
import argparse
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from tqdm.auto import tqdm

from models import get_models
from diffusion import create_diffusion
from utils.nanowm_utils import find_model
from diffusion.df_sample import dfot_sample
from wm_datasets import create_train_val_datasets
from sampling_utils import encode_frames, decode_latents, save_video, save_comparison_video, resize_frames

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    latent_size = args.model.image_size // 8
    args.model.latent_size = latent_size
    model = get_models(args).to(device)
    
    if args.ckpt:
        state_dict = find_model(args.ckpt)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        print(f"Loaded checkpoint: {args.ckpt}")
    
    model.eval()
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(device)
    vae.eval()
    vae_precision = getattr(args.experiment.infra, "vae_precision", "fp32")
    
    diffusion = create_diffusion(
        timestep_respacing=str(args.model.num_sampling_steps),
        noise_schedule=args.experiment.diffusion.noise_schedule,
        pred_name=args.experiment.diffusion.pred_name,
        diffusion_steps=args.experiment.diffusion.diffusion_steps,
        snr_gamma=args.experiment.diffusion.snr_gamma,
        zero_terminal_snr=args.experiment.diffusion.zero_terminal_snr,
    )
    
    # Use standard registry to load dataset
    print("Loading validation dataset...")
    dataset_cfg = OmegaConf.to_container(args.dataset, resolve=True)
    dataset_name = dataset_cfg["name"]
    loader_cfg = dataset_cfg["loader"]

    _, dataset = create_train_val_datasets(
        dataset_name=dataset_name,
        num_frames=args.model.num_frames,
        frame_interval=args.dataset.frame_interval,
        image_size=args.model.image_size,
        **loader_cfg,
    )

    # Parameters
    history_length = args.history_length
    rollout_length = args.rollout_length
    num_frames_per_inference = args.model.num_frames
    frame_interval = args.dataset.frame_interval
    batch_size = args.batch_size
    
    # Filter for valid slices that have enough remaining frames for rollout_length
    print(f"Filtering dataset for slices with enough headroom (rollout_length={rollout_length})...")
    if dataset.slice_mode != "exhaustive":
        raise ValueError(f"Rollout requires exhaustive slice_mode, got '{dataset.slice_mode}'")

    valid_slice_indices = []
    for i in range(len(dataset)):
        slice_idx = dataset.slice_indices[i]
        slice_info = dataset.all_slices[slice_idx]
        traj_idx = slice_info.traj_idx
        start_frame = slice_info.start_frame
        total_len = dataset.data_source.get_seq_length(traj_idx)
        if (total_len - start_frame) >= rollout_length * frame_interval:
            valid_slice_indices.append(i)
        if len(valid_slice_indices) >= args.num_samples:
            break
            
    num_samples = len(valid_slice_indices)
    if num_samples < args.num_samples:
        print(f"Warning: Only found {num_samples} valid slices with enough length (requested {args.num_samples})")
    
    print(f"\nRollout configuration:")
    print(f"  Total samples to process: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  History length: {history_length}")
    print(f"  Rollout length: {rollout_length}")
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx
        
        print(f"Processing batch {start_idx//batch_size + 1} (slices {start_idx} to {end_idx-1})...")
        
        batch_metas = []
        batch_visual_list = []
        batch_raw_actions_list = []
        
        for i in range(start_idx, end_idx):
            ds_idx = valid_slice_indices[i]
            si = dataset.slice_indices[ds_idx]
            slice_info = dataset.all_slices[si]
            traj_idx = slice_info.traj_idx
            start_frame = slice_info.start_frame
            batch_metas.append({"traj_idx": traj_idx, "start_idx": start_frame})

            # Load visual frames via DataSource (raw, not resized/normalized)
            end_frame = start_frame + rollout_length * frame_interval
            visual_frames = dataset.data_source.load_visual_frames(
                index=traj_idx,
                start=start_frame,
                end=end_frame,
                step=frame_interval
            )  # [T, C, H, W] in [0, 1]

            # Resize to match training resolution
            visual_frames = resize_frames(visual_frames, dataset.image_size, dataset.resize_mode)

            # Normalize pixels to [-1, 1]
            visual_frames = visual_frames * 2.0 - 1.0

            # Load raw actions from trajectory
            traj_data = dataset.data_source.load_trajectory(traj_idx)
            raw_actions = traj_data.actions[start_frame:end_frame].clone()

            # Normalize actions if the dataset was configured to do so
            if dataset.normalize_action:
                raw_actions = (raw_actions - dataset.action_mean) / dataset.action_std

            batch_visual_list.append(visual_frames)
            batch_raw_actions_list.append(raw_actions)
            
        gt_visual = torch.stack(batch_visual_list, dim=0).to(device)
        batch_raw_actions = torch.stack(batch_raw_actions_list, dim=0).to(device) 
        
        gt_latents = encode_frames(vae, gt_visual, device, vae_precision=vae_precision)
        generated_latents = gt_latents[:, :history_length]
        
        # Now every sample in the batch has exactly rollout_length frames
        for t in tqdm(range(history_length, rollout_length), desc=f"Rollout progress"):
            context_latents = generated_latents[:, -history_length:]
            
            start_frame_idx = t - history_length
            end_frame_idx = start_frame_idx + num_frames_per_inference
            start_raw_idx = start_frame_idx * frame_interval
            end_raw_idx = end_frame_idx * frame_interval
            
            if end_raw_idx > batch_raw_actions.shape[1]:
                pad_len = end_raw_idx - batch_raw_actions.shape[1]
                last_act = batch_raw_actions[:, -1:]
                window_raw_actions = torch.cat([batch_raw_actions[:, start_raw_idx:], last_act.repeat(1, pad_len, 1)], dim=1)
            else:
                window_raw_actions = batch_raw_actions[:, start_raw_idx:end_raw_idx]
            
            window_actions = rearrange(window_raw_actions, "b (n f) d -> b n (f d)", n=num_frames_per_inference)
            
            model_kwargs = dict(y=None, use_fp16=args.use_fp16)
            if args.model.use_action:
                model_kwargs["action"] = window_actions
            
            shape = (current_batch_size, num_frames_per_inference, 4, latent_size, latent_size)
            pred_latents = dfot_sample(
                diffusion=diffusion,
                model=model.forward,
                shape=shape,
                context=context_latents,
                n_context_frames=history_length,
                scheduling_mode=args.model.scheduling_mode,
                num_sampling_steps=args.model.num_sampling_steps,
                model_kwargs=model_kwargs,
                device=device,
                progress=False,
                eta=args.eta,
                clip_denoised=False,
                n_generate_frames=1,
                history_stabilization_level=args.experiment.diffusion.history_stabilization_level,
            )
            
            new_latent = pred_latents[:, history_length:history_length+1]
            generated_latents = torch.cat([generated_latents, new_latent], dim=1)
            
        print(f"Decoding and saving batch results...")
        gen_frames_batch = decode_latents(vae, generated_latents, vae_precision=vae_precision)
        gt_frames_batch = decode_latents(vae, gt_latents, vae_precision=vae_precision)
        
        for i in range(current_batch_size):
            sample_id = start_idx + i
            save_video(gen_frames_batch[i], str(save_dir / f"sample_{sample_id:04d}_gen.mp4"), fps=args.fps)
            save_video(gt_frames_batch[i], str(save_dir / f"sample_{sample_id:04d}_gt.mp4"), fps=args.fps)
            save_comparison_video(gt_frames_batch[i], gen_frames_batch[i], str(save_dir / f"sample_{sample_id:04d}_compare.mp4"), fps=args.fps)
            
    print(f"Done! Processed {num_samples} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Long-term rollout with Diffusion Forcing using Sliced Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--save_path", type=str, default="rollout_results", help="Directory to save videos")
    parser.add_argument("--num_samples", type=int, default=1, help="Total number of samples")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for parallel rollout")
    parser.add_argument("--rollout_length", type=int, default=50, help="Rollout frames")
    parser.add_argument("--history_length", type=int, default=4, help="Context frames")
    parser.add_argument("--scheduling_mode", type=str, default="sequential",
                        choices=["full_sequence", "pyramid", "sequential"],
                        help="Scheduling mode")
    parser.add_argument("--num_sampling_steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--fps", type=int, default=8, help="FPS")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--history_stabilization_level", type=float, default=None,
                        help="DFoT stabilization noise level in [0, 1) for context frames. "
                             "0.0 disables; 0.02 is the DFoT default recommended starting point.")
    parser.add_argument("--use_fp16", action="store_true", help="Use fp16")
    parser.add_argument("--vae_model_path", type=str, default=None,
                        help="Override the VAE path from the training config (e.g. local dir for offline nodes).")

    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    # Map CLI args to their correct positions in the hierarchy
    cli_args = vars(args)
    cli_overrides = {}

    model_keys = {'scheduling_mode', 'num_sampling_steps'}
    for key in model_keys:
        if cli_args.get(key) is not None:
            cli_overrides.setdefault('model', {})[key] = cli_args[key]

    # CLI override: history_stabilization_level lives under experiment.diffusion.
    if cli_args.get('history_stabilization_level') is not None:
        cli_overrides.setdefault('experiment', {}).setdefault('diffusion', {})[
            'history_stabilization_level'
        ] = cli_args['history_stabilization_level']

    # Top-level / non-hierarchical CLI args
    for key in ('ckpt', 'save_path', 'num_samples', 'batch_size', 'rollout_length',
                'history_length', 'fps', 'eta', 'use_fp16', 'vae_model_path'):
        val = cli_args.get(key)
        if val is not None:
            cli_overrides[key] = val

    if cli_overrides:
        conf = OmegaConf.merge(conf, OmegaConf.create(cli_overrides))

    if not conf.get('ckpt'):
        raise ValueError("Checkpoint path required.")

    main(conf)
