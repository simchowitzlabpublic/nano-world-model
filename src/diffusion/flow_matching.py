"""
Flow Matching for NanoWM — drop-in replacement for GaussianDiffusion.

Implements conditional flow matching (Lipman et al. 2022 / Liu et al. 2022)
with a linear interpolation path:

    Forward:  x_t = τ * x_0 + (1 - τ) * ε,   τ = 1 - (t + 1) / T ∈ [0, 1)
    Target:   u = x_0 - ε                       (constant vector field)
    Loss:     MSE(model(x_t, t), u)

Inference uses simple Euler integration of the learned ODE:

    x_{t-1} = x_t + (τ_{t-1} - τ_t) * u_θ(x_t, t)

Key differences from GaussianDiffusion:
  - No β schedule; noise level is purely linear in τ
  - Training target is always u = x_0 - ε (no SNR weighting needed)
  - Inference needs far fewer steps (~10-20 vs 50-250 for DDIM)
  - Same dfot_sample_loop / dfot_ddim_sample API so df_sample.py is unchanged

API is intentionally identical to GaussianDiffusion so the training loop,
sampling utilities, and dfot_sample() all work without modification.

OT-CFM (mini-batch optimal transport, Tong et al. 2023)
-------------------------------------------------------
Standard FM samples (x_0, ε) independently, which produces crossing paths in
latent space — conflicting gradients that force the model to learn a curved
vector field.  OT-CFM instead solves a mini-batch assignment problem:

    σ* = argmin_σ  Σᵢ ‖x_0ⁱ − ε^σ(ⁱ)‖²

pairing each data point with the noise sample it is geometrically closest to.
This minimises path crossings within the batch, yielding straighter trajectories
and a simpler vector field.  Empirically this requires fewer inference steps
and converges faster.  Opt-in; set use_ot_coupling=True to enable.
"""

import enum
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class PredName(enum.Enum):
    """Routing sentinel used by create_diffusion() to select FlowMatching."""
    FLOW = "flow"


def _mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def _ot_permute(x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Mini-batch optimal transport: permute noise to minimise total squared
    distance to x_start (Tong et al. 2023, OT-CFM).

    Uses the Hungarian algorithm on the [B, B] cost matrix.  The operation is
    O(B³) but runs on CPU with float32 so it is fast for typical batch sizes
    (B ≤ 64).  Gradients are not needed here — this is a data re-ordering step.
    """
    from scipy.optimize import linear_sum_assignment

    B = x_start.shape[0]
    if B == 1:
        return noise  # nothing to permute

    x_flat = x_start.detach().reshape(B, -1).float()
    n_flat = noise.detach().reshape(B, -1).float()
    cost = torch.cdist(x_flat, n_flat, p=2).pow(2).cpu().numpy()
    _, col_ind = linear_sum_assignment(cost)
    return noise[torch.from_numpy(col_ind).to(noise.device)]


def _t_to_tau(t: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """
    Map integer timestep t ∈ [0, T-1] to continuous flow time τ ∈ [0, 1).

    Matches GaussianDiffusion's convention where t=T-1 is the noisiest and
    t=0 is the least noisy (t=-1 = fully clean, handled upstream).

    Flow matching convention:
      τ = 0  →  x_t = ε  (pure noise)
      τ = 1  →  x_t = x_0 (pure data)

    Mapping (so denoising from high t to low t = integrating τ from 0→1):
      t = T-1  → τ = 0       (noisiest diffusion t → pure noise in FM)
      t = 0    → τ = 1-1/T ≈ 1 (least noisy diffusion t → nearly clean in FM)
    """
    return 1.0 - (t.float() + 1.0) / num_timesteps


def _broadcast_tau(tau: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand τ to broadcast against a [B, ...] or [B, F, ...] target."""
    while len(tau.shape) < len(target.shape):
        tau = tau[..., None]
    return tau


class FlowMatching:
    """
    Conditional flow matching world model — same interface as GaussianDiffusion.

    Parameters
    ----------
    num_timesteps : int
        Number of discrete timestep levels (default 1000, same as diffusion).
        Kept large so the existing scheduling matrices / logit-normal sampler
        work without modification.
    snr_gamma : float
        Ignored (kept for API parity). Flow matching uses uniform loss weighting.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        snr_gamma: float = 0.0,
        use_ot_coupling: bool = False,
    ):
        self.num_timesteps = num_timesteps
        self.snr_gamma = snr_gamma  # unused, kept for API compat
        self.use_ot_coupling = use_ot_coupling

        # Dummy betas array so any code that reads diffusion.betas doesn't crash.
        # All entries = 1/T so the cumulative product decays linearly, mirroring
        # the linear interpolation path.
        self.betas = np.full(num_timesteps, 1.0 / num_timesteps, dtype=np.float64)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t from the forward process: x_t = τ * x_0 + (1 - τ) * ε.

        Signature matches GaussianDiffusion.q_sample.  Returns x_t only
        (not the noise) to stay compatible with dfot_sample() which calls
        this as `context = diffusion.q_sample(context, t_stab)`.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        tau = _t_to_tau(t, self.num_timesteps)
        tau = _broadcast_tau(tau, x_start)
        return tau * x_start + (1.0 - tau) * noise

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Optional[dict] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute flow matching training loss for one batch.

        Returns a dict with keys "mse" and "loss" (identical for FM since
        we don't apply SNR weighting).  Shape matches GaussianDiffusion so
        the training loop can call `.mean()` on loss["loss"] unchanged.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        if self.use_ot_coupling:
            noise = _ot_permute(x_start, noise)

        x_t = self.q_sample(x_start, t, noise=noise)
        target = x_start - noise  # u = x_0 - ε

        model_output = model(x_t, t, **model_kwargs)
        assert model_output.shape == target.shape == x_start.shape

        mse = _mean_flat((target - model_output) ** 2)
        return {"mse": mse, "loss": mse}

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _predict_xstart(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover x_0 from x_t and the predicted vector field u_θ.

        Derivation:
            x_t = τ * x_0 + (1 - τ) * ε
            u   = x_0 - ε
          ⟹ x_0 = x_t + (1 - τ) * u
        """
        tau = _t_to_tau(t, self.num_timesteps)
        tau = _broadcast_tau(tau, x_t)
        return x_t + (1.0 - tau) * u

    # ------------------------------------------------------------------
    # DFoT sampling (same API as GaussianDiffusion)
    # ------------------------------------------------------------------

    def dfot_ddim_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        curr_t: torch.Tensor,
        next_t: torch.Tensor,
        clip_denoised: bool = True,
        denoised_fn=None,
        model_kwargs: Optional[dict] = None,
        eta: float = 0.0,          # unused for FM (deterministic ODE)
    ) -> dict:
        """
        One Euler ODE step for flow matching with per-frame timesteps.

        Replaces GaussianDiffusion.dfot_ddim_sample.  Arguments and return
        value are identical so dfot_sample_loop works without changes.

        curr_t, next_t : [B, F] integer timesteps, -1 = already clean.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if denoised_fn is not None:
            raise NotImplementedError("FlowMatching does not support denoised_fn.")
        if clip_denoised:
            raise NotImplementedError(
                "FlowMatching does not support clip_denoised=True; pass "
                "clip_denoised=False."
            )

        curr_t_clamped = curr_t.clamp(min=0)

        # Predict vector field u_θ(x_t, t)
        u = model(x, curr_t_clamped, **model_kwargs)

        # Recover x_0 estimate (for logging / clip)
        pred_xstart = self._predict_xstart(x, curr_t_clamped, u)

        # Continuous time values
        tau_curr = _t_to_tau(curr_t_clamped, self.num_timesteps)   # [B, F]

        next_t_clamped = next_t.clamp(min=0)
        tau_next = _t_to_tau(next_t_clamped, self.num_timesteps)   # [B, F]

        # Where next_t == -1 (fully denoised), τ_next = 1.0 (fully clean in FM convention:
        # τ=0 = pure noise, τ=1 = pure data, since τ = 1 - (t+1)/T).
        # The old value of 0.0 was wrong — it made the last Euler step jump backward
        # from τ≈0.98 to τ=0 (noise), undoing all the denoising progress.
        clean_mask = (next_t < 0)                                   # [B, F]
        tau_next = tau_next.masked_fill(clean_mask, 1.0)

        # Broadcast to [B, F, C, H, W]
        tau_curr_b = _broadcast_tau(tau_curr, x)
        tau_next_b = _broadcast_tau(tau_next, x)

        # Euler step: x_{t-1} = x_t + (τ_next - τ_curr) * u
        sample = x + (tau_next_b - tau_curr_b) * u

        # No update when curr_t == next_t (already at target noise level)
        no_change = (curr_t == next_t)[..., None, None, None].expand_as(sample)
        sample = torch.where(no_change, x, sample)

        # No update when curr_t == -1 (already clean)
        already_clean = (curr_t < 0)[..., None, None, None].expand_as(sample)
        sample = torch.where(already_clean, x, sample)

        return {"sample": sample, "pred_xstart": pred_xstart}

    def dfot_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        scheduling_matrix: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        n_context_frames: int = 0,
        clip_denoised: bool = True,
        denoised_fn=None,
        model_kwargs: Optional[dict] = None,
        device=None,
        progress: bool = False,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate video frames using Euler integration over the flow ODE.

        Identical signature to GaussianDiffusion.dfot_sample_loop.
        scheduling_matrix : [num_steps, F] integer timesteps (-1 = clean).
        """
        if device is None:
            device = next(model.parameters()).device

        batch_size, num_frames = shape[:2]

        # Start from pure noise
        img = torch.randn(*shape, device=device)

        # Overwrite context frames with (possibly lightly-noised) observations
        if context is not None and n_context_frames > 0:
            img[:, :n_context_frames] = context[:, :n_context_frames]

        # Context mask: 1 = keep as-is across steps
        context_mask = torch.zeros(batch_size, num_frames, dtype=torch.long, device=device)
        if n_context_frames > 0:
            context_mask[:, :n_context_frames] = 1

        scheduling_matrix = scheduling_matrix.to(device)
        if scheduling_matrix.dim() == 2:
            scheduling_matrix = scheduling_matrix.unsqueeze(1).expand(-1, batch_size, -1)

        num_steps = scheduling_matrix.shape[0] - 1

        iterator = range(num_steps)
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="Flow Matching Sampling")

        for step in iterator:
            curr_t = scheduling_matrix[step]       # [B, F]
            next_t = scheduling_matrix[step + 1]   # [B, F]

            img_prev = img.clone()

            with torch.no_grad():
                out = self.dfot_ddim_sample(
                    model,
                    img,
                    curr_t,
                    next_t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                img = out["sample"]

            # Restore context frames (they must not drift)
            ctx_expanded = context_mask.view(batch_size, num_frames, *([1] * (len(shape) - 2)))
            img = torch.where(ctx_expanded >= 1, img_prev, img)

        return img
