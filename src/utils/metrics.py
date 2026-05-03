import torch
import torch.nn as nn
import lpips
import piqa
from einops import rearrange
from utils.fvd import compute_fvd
import numpy as np
import scipy.linalg
import torch
from utils.utils import resize_video
import torchvision
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torchvision import transforms
from PIL import Image
import time
import os
# import dreamsim

def batch_forward(batch_size, input1, input2, forward):
    assert input1.shape[0] == input2.shape[0]
    first_result = forward(input1[0:1], input2[0:1])
    is_tuple = isinstance(first_result, tuple)

    if is_tuple:
        results = [[] for _ in range(len(first_result))]
    else:
        results = []

    for i in range(0, input1.shape[0], batch_size):
        result = forward(input1[i: i + batch_size], input2[i: i + batch_size])
        if is_tuple:
            for j in range(len(result)):
                if isinstance(result[j], torch.Tensor):
                    results[j].append(result[j].detach())
                else:
                    results[j].append(result[j])
        else:
            if isinstance(result, torch.Tensor):
                results.append(result.detach())
            else:
                results.append(result)

    if is_tuple:
        return tuple(np.concatenate(r, axis=0) for r in results)
    else:
        return torch.cat(results, dim=0)

import cv2

def save_video_h264(video_np, save_path, fps=3):
    # video_np: [T, H, W, C], RGB, uint8
    # Prefer H.264 for browser/W&B compatibility; fall back to mp4v if needed.
    try:
        frames = torch.from_numpy(video_np)
        if frames.dtype != torch.uint8:
            frames = frames.to(torch.uint8)
        torchvision.io.write_video(
            save_path,
            frames,
            fps=fps,
            video_codec="h264",
            options={"crf": "10"},
        )
        return
    except Exception:
        pass

    h, w = video_np.shape[1], video_np.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for frame in video_np:
        # RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

class Evaluator():
    def __init__(self, i3d_model_path=None, max_batchsize=2, device='cuda:0', env=None, save_dir=None):
        """
        Initialize evaluator with optional I3D model for FVD computation.
        """
        self.device = device
        self.max_batchsize = max_batchsize
        self.i3d_model_path = i3d_model_path
        self.env = env
        self.mse_loss = nn.MSELoss(reduction='none').to(self.device)

        # input region: [0, 1] 
        # Note: actually, as long as the input range [0, a], the calculated psnr is the same, however, for code consistency, we use [0, 1]
        self.psnr_metric = piqa.PSNR(epsilon=1e-08, value_range=1.0, reduction='none').to(self.device).eval()

        # 11x11 Gaussian window, sigma=1.5, n_channels=3 (common paper configuration)
        # Input range required: [0, 1]
        self.ssim_metric = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3, reduction='none').to(self.device).eval()

        # Input range required: [-1, 1]
        # For recon environments, use alexnet; for others, use vgg;
        # This is for following NWM evaluation setup
        lpips_net = 'alex' if env in ['recon_time', 'recon_rollout'] else 'vgg'
        self.lpips_model = lpips.LPIPS(net=lpips_net).to(self.device).eval()

        # DreamSim model with its official preprocessing
        # Use home directory cache instead of polluting project directory
        # dreamsim_cache = os.path.expanduser("~/.cache/dreamsim")
        # self.dreamsim_model, self.dreamsim_preprocess = dreamsim.dreamsim(
        #     pretrained=True, device=self.device, cache_dir=dreamsim_cache
        # )
        # self.dreamsim_model.eval()

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.iv3_model = InceptionV3([block_idx]).to(self.device).eval()
        self.save_dir = save_dir
        self.save_video = (save_dir is not None)

    def compute_mse(self, x_pred: torch.Tensor, x_gt: torch.Tensor) -> float:
        if self.max_batchsize is not None and x_pred.shape[0] > self.max_batchsize:
            mse_val = batch_forward(
                batch_size=self.max_batchsize,
                input1=x_pred,
                input2=x_gt,
                forward=lambda a, b: self.mse_loss(a, b),
            )
        else:
            mse_val = self.mse_loss(x_pred, x_gt)
        return mse_val.mean().item()
    def compute_psnr(self, x_pred: torch.Tensor, x_gt: torch.Tensor) -> float:
        x_pred = (x_pred + 1.0) / 2.0
        x_gt = (x_gt + 1.0) / 2.0
        if self.max_batchsize is not None and x_pred.shape[0] > self.max_batchsize:
            psnr_val = batch_forward(
                batch_size=self.max_batchsize,
                input1=x_pred,
                input2=x_gt,
                forward=lambda a, b: self.psnr_metric(a, b),
            )
        else:
            psnr_val = self.psnr_metric(x_pred, x_gt)
        return psnr_val.mean().item()

    def compute_ssim(self, x_pred: torch.Tensor, x_gt: torch.Tensor) -> float:
        x_pred = (x_pred + 1.0) / 2.0
        x_gt = (x_gt + 1.0) / 2.0
        # x_pred = rearrange(x_pred, '(b t) c h w -> b c t h w', b=self.B)
        # x_gt = rearrange(x_gt, '(b t) c h w -> b c t h w', b=self.B)
        if self.max_batchsize is not None and x_pred.shape[0] > self.max_batchsize:
            ssim_val = batch_forward(
                batch_size=self.max_batchsize,
                input1=x_pred,
                input2=x_gt,
                forward=lambda a, b: self.ssim_metric(a, b),
            )
        else:
            ssim_val = self.ssim_metric(x_pred, x_gt)
        return ssim_val.mean().item()
    def compute_lpips(self, x_pred: torch.Tensor, x_gt: torch.Tensor) -> float:
        if self.max_batchsize is not None and x_pred.shape[0] > self.max_batchsize:
            lpips_val = batch_forward(
                batch_size=self.max_batchsize,
                input1=x_pred,
                input2=x_gt,
                forward=lambda a, b: self.lpips_model(a, b),
            )
        else:
            lpips_val = self.lpips_model(x_pred, x_gt)
        return lpips_val.mean().item()

    # def compute_dreamsim(self, x_pred: torch.Tensor, x_gt: torch.Tensor) -> float:
    #     # Convert from [-1, 1] to [0, 1]
    #     x_pred = (x_pred + 1.0) / 2.0
    #     x_gt = (x_gt + 1.0) / 2.0
        
    #     # Apply DreamSim's official preprocessing
    #     def preprocess_and_compute(a, b):
    #         # Convert Tensor to PIL Image before preprocessing
    #         # Tensor shape: [B, C, H, W], value range: [0, 1]
    #         def tensor_to_pil(tensor_img):
    #             # tensor_img: [C, H, W]
    #             # Convert to [H, W, C] and then to numpy
    #             img_np = tensor_img.permute(1, 2, 0).cpu().numpy()
    #             # Convert from [0, 1] to [0, 255] and uint8
    #             img_np = (img_np * 255).astype(np.uint8)
    #             # Convert to PIL Image
    #             return Image.fromarray(img_np)
            
    #         # DreamSim preprocess expects PIL Images, process in batch
    #         a_list = [self.dreamsim_preprocess(tensor_to_pil(a[i])) for i in range(a.shape[0])]
    #         b_list = [self.dreamsim_preprocess(tensor_to_pil(b[i])) for i in range(b.shape[0])]
    #         a_processed = torch.cat(a_list, dim=0).to(self.device)
    #         b_processed = torch.cat(b_list, dim=0).to(self.device)
    #         return self.dreamsim_model(a_processed, b_processed)
        
    #     if self.max_batchsize is not None and x_pred.shape[0] > self.max_batchsize:
    #         dreamsim_val = batch_forward(
    #             batch_size=self.max_batchsize,
    #             input1=x_pred,
    #             input2=x_gt,
    #             forward=preprocess_and_compute,
    #         )
    #     else:
    #         dreamsim_val = preprocess_and_compute(x_pred, x_gt)
    #     return dreamsim_val.mean().item()

    def _compute_fvd(self, x_pred: torch.Tensor, x_gt: torch.Tensor, raw: bool = False) -> float:
        # Upstream `evaluate_all` clamps inputs to [-1, 1]. i3d expects [0, 1], so convert below.
        x_pred = rearrange(x_pred, '(b t) c h w -> b t h w c', b=self.B)
        x_gt = rearrange(x_gt, '(b t) c h w -> b t h w c', b=self.B)
        x_pred = resize_video(x_pred, (224, 224))
        x_gt = resize_video(x_gt, (224, 224))
        x_pred = (x_pred + 1.0) / 2.0
        x_gt = (x_gt + 1.0) / 2.0
        x_pred=rearrange(x_pred, 'b t h w c -> b c t h w')
        x_gt=rearrange(x_gt, 'b t h w c -> b c t h w')
        # [B, C, T, H, W]
        # FVD's mu & sigma relies on dataset size, therefore we store the features across batch, and compute the mu & sigma in the end
        if self.max_batchsize is not None and x_pred.shape[0] > self.max_batchsize:
            raw_features_true, raw_features_pred = batch_forward(
                batch_size=self.max_batchsize,
                input1=x_pred,
                input2=x_gt,
                forward=lambda a, b: compute_fvd(a, b, device=self.device, max_items=None, batch_size=8, return_raw_features=True, i3d_model_path=self.i3d_model_path),
            )
        else:
            raw_features_true, raw_features_pred = compute_fvd(
                y_true=x_gt,
                y_pred=x_pred,
                device=self.device,
                max_items=None,  # No limit on number of videos contributing to FVD
                batch_size=2,    # Process two videos at a time for feature computation
                return_raw_features=True,
                i3d_model_path=self.i3d_model_path,
            )
        if raw:
            return raw_features_true, raw_features_pred
        else:
            mu_true = np.mean(raw_features_true, axis=0)
            mu_pred = np.mean(raw_features_pred, axis=0)
            sigma_true = np.cov(raw_features_true, rowvar=False)
            sigma_pred = np.cov(raw_features_pred, rowvar=False)
            m = np.square(mu_pred - mu_true).sum()
            _s = scipy.linalg.sqrtm(np.dot(sigma_pred, sigma_true), disp=False)
            s = _s[0] if isinstance(_s, tuple) else _s
            fvd = np.real(m + np.trace(sigma_pred + sigma_true - s * 2))
            return fvd
        
    def _compute_fid(self, x_pred: torch.Tensor, x_gt: torch.Tensor, raw: bool = False) -> float:
        x_pred = rearrange(x_pred, '(b t) c h w -> b t h w c', b=self.B)
        self.T = x_pred.shape[1]
        x_gt = rearrange(x_gt, '(b t) c h w -> b t h w c', b=self.B)
        x_pred = resize_video(x_pred, (299, 299))
        x_gt = resize_video(x_gt, (299, 299))
        x_pred = (x_pred + 1.0) / 2.0
        x_gt = (x_gt + 1.0) / 2.0
        x_pred = rearrange(x_pred, 'b t h w c -> (b t) c h w')
        x_gt = rearrange(x_gt, 'b t h w c -> (b t) c h w')
        # [(b t) c h w], [0, 1]
        real_stats = []
        fake_stats = []

        batchsize = self.max_batchsize if self.max_batchsize is not None else 2
        batchsize*=self.T
        for i in range(0, x_gt.shape[0], batchsize):
            real_batch = x_gt[i : i + batchsize].to(self.device)
            pred_batch = x_pred[i : i + batchsize].to(self.device)

            with torch.no_grad():
                output_real = self.iv3_model(real_batch)[0]  # pylint: disable=E1102
                output_fake = self.iv3_model(pred_batch)[0]  # pylint: disable=E1102
                # [2 2048 1 1]

            output_real = output_real.squeeze(3).squeeze(2).cpu().numpy()   # [2 2048]
            output_fake = output_fake.squeeze(3).squeeze(2).cpu().numpy()
            real_stats.append(output_real)
            fake_stats.append(output_fake)
        
        real_stats = np.concatenate(real_stats, axis=0)
        fake_stats = np.concatenate(fake_stats, axis=0)

        if raw:
            return real_stats, fake_stats
        else:
            mu_real = np.mean(real_stats, axis=0)
            mu_fake = np.mean(fake_stats, axis=0)
            sigma_real = np.cov(real_stats, rowvar=False)
            sigma_fake = np.cov(fake_stats, rowvar=False)
            fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
            return fid_value

    def evaluate_all(self, x_pred: torch.Tensor, x_gt: torch.Tensor, raw: bool, path_dict: dict = None, evaluate: bool=True, compute_fvd: bool = True, n_context_frames: int = 0) -> dict:
        """
        Compute all metrics and return a dictionary.
        Input x_pred and x_gt shapes can be:
          - Image: [B, C, H, W]
          - Video: [B, C, T, H, W]

        IMPORTANT: Metrics are computed ONLY on predicted frames (excluding context frames).
        This aligns with DINO-WM's evaluation protocol where metrics reflect model's
        prediction capability, not reconstruction quality of context frames.

        However, complete videos (including context frames) are still saved for visualization.

        Args:
            x_pred: Predicted video [B, C, T, H, W] (includes context + predicted frames)
            x_gt: Ground truth video [B, C, T, H, W]
            raw: Whether to return raw features for FVD/FID computation
            path_dict: Optional paths for saving videos
            evaluate: Whether to compute metrics
            compute_fvd: Whether to compute FVD metrics (default: True)
            n_context_frames: Number of context frames to exclude from metric computation (default: 0)
                             If > 0, only frames [n_context_frames:] are used for metrics.
        """
        if x_pred.max() > 1 or x_pred.min() < -1:
            x_pred = x_pred.clamp(-1, 1)
        if x_gt.max() > 1 or x_gt.min() < -1:
            x_gt = x_gt.clamp(-1, 1)

        num_frames = x_pred.shape[2] if x_pred.ndim == 5 else 1
        if compute_fvd and num_frames < 10:
            # print(f"[Metrics] Warning: Video length {num_frames} is too short for FVD calculation (requires >= 10). Skipping FVD.")
            compute_fvd = False

        # Save complete videos (including context frames) for visualization
        # region: [-1,1]
        save_video = self.save_video  # Set to True to save videos
        if save_video and x_pred.ndim == 5:
            self.B = x_pred.shape[0]
            # Save complete videos (no slicing)
            save_pred_complete = x_pred.clone()
            save_gt_complete = x_gt.clone()

            # Flatten for per-frame processing
            save_pred_flat = rearrange(save_pred_complete, 'b c t h w -> (b t) c h w')
            save_gt_flat = rearrange(save_gt_complete, 'b c t h w -> (b t) c h w')

            # Convert back to [0, 1] range for saving
            save_pred_flat = (save_pred_flat + 1.0) / 2.0
            save_gt_flat = (save_gt_flat + 1.0) / 2.0

            # Reshape back to video format [B, C, T, H, W]
            save_pred = rearrange(save_pred_flat, '(b t) c h w -> b c t h w', b=self.B)
            save_gt = rearrange(save_gt_flat, '(b t) c h w -> b c t h w', b=self.B)

            for idx in range(save_pred.shape[0]):
                if path_dict is not None:
                    # path_dict[idx] is a string:  e.g. '/home/NAS/rl_data/frame_action_datasets/fractal20220817_data/train_eps_00064664.npz'
                    # here we get the .npz file name
                    npz_file_name = path_dict[idx].split('/')[-1].split('.')[0] if '/' in path_dict[idx] else path_dict[idx]
                    save_folder = f'{self.save_dir}/{npz_file_name}'
                    # Ensure the directory exists before saving video
                    os.makedirs(save_folder, exist_ok=True)
                    # Save complete videos with H.264 when available for browser/W&B compatibility
                    save_video_h264(
                        (save_pred[idx].permute(1, 2, 3, 0) * 255).cpu().numpy().astype(np.uint8),
                        f'{save_folder}/pred_video.mp4',
                        fps=3
                    )
                    save_video_h264(
                        (save_gt[idx].permute(1, 2, 3, 0) * 255).cpu().numpy().astype(np.uint8),
                        f'{save_folder}/gt_video.mp4',
                        fps=3
                    )
                    # Save raw data as .npz
                    np.savez(
                        f'{save_folder}/raw_data.npz',
                        pred=save_pred[idx].cpu().numpy(),
                        gt=save_gt[idx].cpu().numpy()
                    )

        # Compute metrics ONLY on predicted frames (exclude context frames)
        # This aligns with DINO-WM protocol: evaluate prediction quality, not context reconstruction
        if x_pred.ndim == 5 and n_context_frames > 0:
            # Slice out context frames for metric computation
            x_pred_for_metrics = x_pred[:, :, n_context_frames:, :, :]
            x_gt_for_metrics = x_gt[:, :, n_context_frames:, :, :]

            # Update num_frames for FVD check
            num_frames_for_metrics = x_pred_for_metrics.shape[2]
            if compute_fvd and num_frames_for_metrics < 10:
                compute_fvd = False

            # Flatten for metric computation
            self.B = x_pred_for_metrics.shape[0]
            x_pred = rearrange(x_pred_for_metrics, 'b c t h w -> (b t) c h w')
            x_gt = rearrange(x_gt_for_metrics, 'b c t h w -> (b t) c h w')
        elif x_pred.ndim == 5:
            # No context frames to exclude, compute on all frames
            self.B = x_pred.shape[0]
            x_pred = rearrange(x_pred, 'b c t h w -> (b t) c h w')
            x_gt = rearrange(x_gt, 'b c t h w -> (b t) c h w')

        if not evaluate:
            return None

        x_pred = x_pred.to(self.device)
        x_gt = x_gt.to(self.device)

        results = {}
        results['mse'] = self.compute_mse(x_pred, x_gt)
        results['psnr'] = self.compute_psnr(x_pred, x_gt)
        results['ssim'] = self.compute_ssim(x_pred, x_gt)
        results['lpips'] = self.compute_lpips(x_pred, x_gt)
        # results['dreamsim'] = self.compute_dreamsim(x_pred, x_gt)
        if raw:
            if compute_fvd:
                results['raw_gt_features'], results['raw_pred_features'] = self._compute_fvd(x_pred, x_gt, raw=True)
            results['real_stats'], results['fake_stats'] = self._compute_fid(x_pred, x_gt, raw=True)
        else:
            if compute_fvd:
                results['fvd'] = self._compute_fvd(x_pred, x_gt, raw=False)
            results['fid'] = self._compute_fid(x_pred, x_gt, raw=False)
        return results

