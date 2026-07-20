# Callback functions for pytorch lightning training
# see train_pl.py
# Modified from: https://github.com/thuml/Vid2World/blob/main/main/callbacks.py

import os
import time
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from utils.save_video import log_local, prepare_to_log, tensor_to_mp4
from utils.metrics import Evaluator
import numpy as np
import scipy

from pytorch_fid.fid_score import calculate_frechet_distance



class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    @staticmethod
    def _gpu_index(trainer):
        # Lightning 2.x API: root_gpu / training_type_plugin were removed in 2.0.
        return trainer.strategy.root_device.index

    def on_train_epoch_start(self, trainer, pl_module):
        gpu_index = self._gpu_index(trainer)
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        gpu_index = self._gpu_index(trainer)
        torch.cuda.synchronize(gpu_index)
        max_memory = torch.cuda.max_memory_allocated(gpu_index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            # Lightning 2.x: trainer.strategy (training_type_plugin was removed).
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
    
class MetricsLogger(Callback):
    def __init__(self, env=None, log_every_n_train_steps=1000, buffer_size=32, i3d_model_path=None, max_batchsize=2, log_images_kwargs=None, save_dir=None, evaluate=True):
        """
        evaluator: see utils.metrics.py
        log_every_n_train_steps: log every n train steps
        buffer_size: the number of batches to accumulate for calculating metrics, to avoid large fluctuations
        """
        super().__init__()
        self.env = env
        self.log_every_n_train_steps = log_every_n_train_steps
        self.buffer_size = buffer_size
        self.i3d_model_path = i3d_model_path
        self.max_batchsize = max_batchsize
        self.save_dir = save_dir
        self.evaluate = evaluate
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}   
        self.evaluator = Evaluator(
            i3d_model_path=self.i3d_model_path,
            max_batchsize=self.max_batchsize,
            device='cuda:0',
            env=self.env,
            save_dir=self.save_dir
        )
        self.val_metrics_buffer = {}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if not self.evaluate:
            return

        is_val_only = trainer.state.fn == 'validate'
        should_log = is_val_only or (trainer.global_step % self.log_every_n_train_steps == 0)

        if not should_log and not trainer.sanity_checking:
            return

        with torch.no_grad():
            x_pred, x_reconst, x_gt = self.get_predictions(pl_module, batch, split="val")
            # Clamp everything that flows into `.byte()` below: bf16 VAE decode can
            # overshoot [-1, 1] by ~0.01 and uint8 cast wraps mod 256 → speckles.
            x_pred = x_pred.clamp(-1, 1)
            x_reconst = x_reconst.clamp(-1, 1)
            x_gt = x_gt.clamp(-1, 1)

            # Get number of context frames
            n_context = pl_module.args.model.n_context_frames

            paths = None
            if self.save_dir:
                step_dir = os.path.join(self.save_dir, f"step_{trainer.global_step:07d}")
                os.makedirs(step_dir, exist_ok=True)
                self.evaluator.save_dir = step_dir
                paths = []
                for i in range(x_pred.shape[0]):
                    traj_idx = batch['meta_info']['traj_idx'][i].item() if 'meta_info' in batch else i
                    start_idx = batch['meta_info']['start_idx'][i].item() if 'meta_info' in batch else 0
                    paths.append(f"traj_{traj_idx:04d}_start_{start_idx:04d}")

            # Save complete videos (including context frames) for visualization
            # But only compute metrics on predicted frames (excluding context) for fair evaluation
            # This aligns with DINO-WM's evaluation protocol
            metrics = self.evaluator.evaluate_all(
                x_pred, x_gt,
                raw=True,
                path_dict=paths,
                evaluate=self.evaluate,
                n_context_frames=n_context  # Pass context info to evaluator
            )

            # Log complete videos (including context frames) for visualization
            self._log_videos(trainer, pl_module, batch, x_pred, x_reconst, x_gt, batch_idx)
            
            if metrics is not None:
                for key, value in metrics.items():
                    if key in ["raw_gt_features", "raw_pred_features", "real_stats", "fake_stats"]:
                        tensorized_value = torch.tensor(value, device=pl_module.device)
                        if key not in self.val_metrics_buffer.keys():
                            self.val_metrics_buffer[key] = tensorized_value
                        else:
                            self.val_metrics_buffer[key]= torch.cat([self.val_metrics_buffer[key], tensorized_value], dim=0)
                    else:
                        if key not in self.val_metrics_buffer.keys():
                            self.val_metrics_buffer[key] = torch.tensor(value, device=pl_module.device).unsqueeze(0)
                        else:
                            self.val_metrics_buffer[key]= torch.cat([self.val_metrics_buffer[key], torch.tensor(value, device=pl_module.device).unsqueeze(0)], dim=0)
            
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.evaluate or not self.val_metrics_buffer:
            return
            
        gather_val_metrics_buffer = pl_module.all_gather(self.val_metrics_buffer)
        
        # Flatten the gathered tensors
        feature_keys = {"raw_gt_features", "raw_pred_features", "real_stats", "fake_stats"}
        for key, value in gather_val_metrics_buffer.items():
            if key in feature_keys:
                gather_val_metrics_buffer[key] = value.reshape(-1, value.shape[-1])
            else:
                gather_val_metrics_buffer[key] = value.reshape(-1)

        with torch.no_grad():
            if pl_module.global_rank == 0:
                mu_pred, mu_true = None, None
                sigma_pred, sigma_true = None, None
                mu_real, mu_fake = None, None
                sigma_real, sigma_fake = None, None

                for key, value in gather_val_metrics_buffer.items():
                    if key == "raw_gt_features":
                        raw_gt_features = value.cpu().numpy()
                        mu_true = np.mean(raw_gt_features, axis=0)
                        sigma_true = np.cov(raw_gt_features, rowvar=False)
                    elif key == "raw_pred_features":
                        raw_pred_features = value.cpu().numpy()
                        mu_pred = np.mean(raw_pred_features, axis=0)
                        sigma_pred = np.cov(raw_pred_features, rowvar=False)
                    elif key == "real_stats":
                        real_stats = value.cpu().numpy()
                        mu_real = np.mean(real_stats, axis=0)
                        sigma_real = np.cov(real_stats, rowvar=False)
                    elif key == "fake_stats":
                        fake_stats = value.cpu().numpy()
                        mu_fake = np.mean(fake_stats, axis=0)
                        sigma_fake = np.cov(fake_stats, rowvar=False)
                    else:
                        mean_value = torch.mean(value)
                        pl_module.log(f"val_eval/{key}", mean_value, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
                        mainlogger.info(f"Epoch end val {key}: {mean_value}")
                
                if mu_pred is not None and mu_true is not None:
                    m = np.square(mu_pred - mu_true).sum()
                    s = scipy.linalg.sqrtm(np.dot(sigma_pred, sigma_true))
                    fvd = np.real(m + np.trace(sigma_pred + sigma_true - s * 2))
                    pl_module.log("val_eval/fvd", fvd, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
                    mainlogger.info(f"Epoch end val fvd: {fvd}")

                if mu_real is not None and mu_fake is not None:
                    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
                    pl_module.log("val_eval/fid", fid, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
                    mainlogger.info(f"Epoch end val fid: {fid}")

            self.val_metrics_buffer = {}
            torch.cuda.empty_cache()

    def _log_videos(self, trainer, pl_module, batch, x_pred, x_reconst, x_gt, batch_idx):
        if not pl_module.logger:
            return

        logger_experiments = []
        if hasattr(pl_module.logger, "experiment"):
            logger_experiments.append(pl_module.logger.experiment)
        if hasattr(pl_module.logger, "loggers"):
            for logger in pl_module.logger.loggers:
                if hasattr(logger, "experiment"):
                    logger_experiments.append(logger.experiment)
        if not logger_experiments:
            return

        n_log = min(x_pred.shape[0], 4)

        n_context = pl_module.args.model.n_context_frames

        for i in range(n_log):
            if 'meta_info' in batch:
                t_idx = batch['meta_info']['traj_idx'][i].item()
                s_idx = batch['meta_info']['start_idx'][i].item()
                sample_tag = f"traj_{t_idx:04d}_start_{s_idx:04d}"
            else:
                sample_tag = f"batch{batch_idx}_idx{i}"

            s_gt = ((x_gt[i] + 1.0) / 2.0).permute(1, 0, 2, 3)    # [T, C, H, W]
            s_pred = ((x_pred[i] + 1.0) / 2.0).permute(1, 0, 2, 3) # [T, C, H, W]
            s_reconst = ((x_reconst[i] + 1.0) / 2.0).permute(1, 0, 2, 3) # [T, C, H, W]
            s_ctx = s_gt[:n_context]                              # [n_ctx, C, H, W]

            for exp in logger_experiments:
                if hasattr(exp, "add_video"):
                    if n_context == 1:
                        exp.add_image(
                            f"val_vis/{sample_tag}/context_frame", s_ctx[0], global_step=trainer.global_step
                        )
                    else:
                        exp.add_video(
                            f"val_vis/{sample_tag}/context_frame", s_ctx.unsqueeze(0), global_step=trainer.global_step, fps=4
                        )

                    exp.add_video(
                        f"val_vis/{sample_tag}/x_gt", s_gt.unsqueeze(0), global_step=trainer.global_step, fps=4
                    )

                    exp.add_video(
                        f"val_vis/{sample_tag}/x_reconst", s_reconst.unsqueeze(0), global_step=trainer.global_step, fps=4
                    )

                    exp.add_video(
                        f"val_vis/{sample_tag}/x_pred", s_pred.unsqueeze(0), global_step=trainer.global_step, fps=4
                    )
                else:
                    try:
                        import wandb
                    except Exception:
                        continue

                    file_tag = None
                    if 'meta_info' in batch:
                        file_tag = f"traj_{t_idx:04d}_start_{s_idx:04d}"
                    use_file_video = False
                    gt_path = None
                    pred_path = None
                    context_path = None
                    reconst_path = None
                    if file_tag and self.evaluator.save_dir:
                        gt_path = os.path.join(self.evaluator.save_dir, file_tag, "gt_video.mp4")
                        pred_path = os.path.join(self.evaluator.save_dir, file_tag, "pred_video.mp4")
                        context_path = os.path.join(self.evaluator.save_dir, file_tag, "context_video.mp4")
                        reconst_path = os.path.join(self.evaluator.save_dir, file_tag, "reconst_video.mp4")
                        use_file_video = os.path.exists(gt_path) and os.path.exists(pred_path)
                        if use_file_video and n_context > 1 and not os.path.exists(context_path):
                            os.makedirs(os.path.dirname(context_path), exist_ok=True)
                            context_frames = (s_ctx.permute(0, 2, 3, 1) * 255).byte().cpu()
                            try:
                                torchvision.io.write_video(
                                    context_path,
                                    context_frames,
                                    fps=3,
                                    video_codec="h264",
                                    options={"crf": "10"},
                                )
                            except Exception:
                                pass
                        if use_file_video and not os.path.exists(reconst_path):
                            os.makedirs(os.path.dirname(reconst_path), exist_ok=True)
                            reconst_frames = (s_reconst.permute(0, 2, 3, 1) * 255).byte().cpu()
                            try:
                                torchvision.io.write_video(
                                    reconst_path,
                                    reconst_frames,
                                    fps=4,
                                    video_codec="h264",
                                    options={"crf": "10"},
                                )
                            except Exception:
                                pass

                    if n_context == 1:
                        ctx_payload = {
                            f"val_vis/{sample_tag}/context_frame": wandb.Image(
                                (s_ctx[0].permute(1, 2, 0) * 255).byte().cpu().numpy()
                            )
                        }
                    else:
                        ctx_payload = {
                            f"val_vis/{sample_tag}/context_frame": (
                                wandb.Video(context_path, format="mp4")
                                if (use_file_video and context_path and os.path.exists(context_path))
                                else wandb.Video(
                                    (s_ctx.permute(0, 2, 3, 1) * 255).byte().cpu().numpy(),
                                    fps=4,
                                    format="mp4",
                                )
                            )
                        }

                    exp.log(
                        {
                            **ctx_payload,
                            f"val_vis/{sample_tag}/x_gt": (
                                wandb.Video(gt_path, format="mp4")
                                if use_file_video
                                else wandb.Video(
                                    (s_gt.permute(0, 2, 3, 1) * 255).byte().cpu().numpy(),
                                    fps=4,
                                    format="mp4",
                                )
                            ),
                            f"val_vis/{sample_tag}/x_reconst": wandb.Video(
                                reconst_path,
                                format="mp4",
                            ) if (use_file_video and reconst_path and os.path.exists(reconst_path)) else wandb.Video(
                                (s_reconst.permute(0, 2, 3, 1) * 255).byte().cpu().numpy(),
                                fps=4,
                                format="mp4",
                            ),
                            f"val_vis/{sample_tag}/x_pred": (
                                wandb.Video(pred_path, format="mp4")
                                if use_file_video
                                else wandb.Video(
                                    (s_pred.permute(0, 2, 3, 1) * 255).byte().cpu().numpy(),
                                    fps=4,
                                    format="mp4",
                                )
                            ),
                        },
                        step=trainer.global_step,
                    )

    def get_predictions(self, pl_module, batch, split="train"):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        
        with torch.no_grad():
            # Use the log_images method we added to NanoWMTrainingModule
            batch_logs = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            
        x_pred = batch_logs["samples"]
        x_reconst = batch_logs["reconst"]
        x_gt = batch_logs["gt"]
        
        if is_train:
            pl_module.train()
        return x_pred, x_reconst, x_gt


class LatentMetricsLogger(Callback):
    def __init__(self, log_every_n_train_steps=1000, log_latents_kwargs=None, evaluate=True):
        super().__init__()
        self.log_every_n_train_steps = log_every_n_train_steps
        self.log_latents_kwargs = log_latents_kwargs if log_latents_kwargs else {}
        self.evaluate = evaluate

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if not self.evaluate:
            return

        is_val_only = trainer.state.fn == "validate"
        should_log = is_val_only or (trainer.global_step % self.log_every_n_train_steps == 0)
        if not should_log and not trainer.sanity_checking:
            return

        was_training = pl_module.training
        if was_training:
            pl_module.eval()

        with torch.no_grad():
            logs = pl_module.log_latents(batch, split="val", **self.log_latents_kwargs)
            z_pred = logs["samples"]
            z_gt = logs["gt"]
            n_context = logs["n_context"]

            pred = z_pred[:, n_context:]
            gt = z_gt[:, n_context:]
            diff = pred.float() - gt.float()

            mse = diff.square().mean()
            mae = diff.abs().mean()
            rmse = torch.sqrt(mse.clamp_min(0.0))
            nmse = mse / gt.float().square().mean().clamp_min(1e-12)
            pred_flat = pred.float().reshape(pred.shape[0], -1)
            gt_flat = gt.float().reshape(gt.shape[0], -1)
            cosine_distance = 1.0 - F.cosine_similarity(pred_flat, gt_flat, dim=1).mean()

            batch_size = pred.shape[0]
            pl_module.log("val_eval/latent_mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
            pl_module.log("val_eval/latent_nmse", nmse, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            pl_module.log("val_eval/latent_mae", mae, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            pl_module.log("val_eval/latent_rmse", rmse, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            pl_module.log("val_eval/latent_cosine_distance", cosine_distance, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)

        if was_training:
            pl_module.train()
