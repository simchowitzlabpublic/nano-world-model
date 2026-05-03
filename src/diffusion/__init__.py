# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import torch

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .flow_matching import FlowMatching


def sample_training_timesteps(
    shape,
    num_timesteps,
    *,
    strategy: str,
    logit_normal_mean: float,
    logit_normal_std: float,
    device=None,
) -> torch.Tensor:
    """Sample discrete training timesteps for the diffusion forward process.

    strategy:
        "uniform"      – torch.randint(0, num_timesteps) (original behavior)
        "logit_normal" – u ~ N(mean, std^2); t = floor(sigmoid(u) * num_timesteps).
                         Proposed in Esser et al. "Scaling RFT / SD3" (2024);
                         concentrates training near the middle-noise regime.
    """
    if strategy == "uniform":
        return torch.randint(0, num_timesteps, shape, device=device)
    if strategy == "logit_normal":
        u = torch.randn(shape, device=device) * logit_normal_std + logit_normal_mean
        t_cont = torch.sigmoid(u)  # (0, 1)
        t = (t_cont * num_timesteps).long().clamp(max=num_timesteps - 1)
        return t
    raise ValueError(
        f"Unknown timestep_sampling strategy: {strategy}. "
        "Use 'uniform' or 'logit_normal'."
    )


def create_diffusion(
    timestep_respacing,
    *,
    noise_schedule,
    pred_name,
    diffusion_steps,
    snr_gamma,
    zero_terminal_snr,
):
    """Build a (possibly respaced) Gaussian diffusion.

    All arguments are required — this is a pipeline entry point, so the config
    is the single source of truth (no silent defaults).

    pred_name: which target the model predicts.
        "epsilon" (alias "eps") — noise
        "x"                      — clean image x_0
        "v"                      — v = sqrt(alpha)*eps - sqrt(1-alpha)*x_0

    zero_terminal_snr: if True, rescale betas so alpha_bar[T] = 0 (Lin et al.
        2023). Recommended to pair with "v" — "epsilon" prediction becomes
        degenerate at the terminal step because x_T is pure noise.
    """
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if zero_terminal_snr:
        betas = gd.enforce_zero_terminal_snr(betas)
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]

    # Flow matching is a separate code path — no beta schedule, no DDIM.
    if pred_name == "flow":
        return FlowMatching(num_timesteps=diffusion_steps, snr_gamma=snr_gamma)

    pred_name_map = {
        "eps": gd.PredName.EPSILON,
        "epsilon": gd.PredName.EPSILON,
        "x": gd.PredName.X,
        "v": gd.PredName.V,
    }
    if pred_name not in pred_name_map:
        raise ValueError(
            f"Unknown pred_name: {pred_name}. Must be one of "
            f"{sorted(set(pred_name_map)) + ['flow']}."
        )

    if pred_name_map[pred_name] == gd.PredName.EPSILON:
        import warnings
        if zero_terminal_snr:
            warnings.warn(
                "zero_terminal_snr=True with pred_name='epsilon' is degenerate at "
                "t=T (x_T is pure noise); use pred_name='v' or 'x'.",
                stacklevel=2,
            )
        elif noise_schedule == "squaredcos_cap_v2":
            warnings.warn(
                "noise_schedule='squaredcos_cap_v2' + pred_name='epsilon' is "
                "numerically pathological: α̅_T ≈ 1e-9 inflates the ε→x0 "
                "reconstruction by ~2e4 and diverges under bf16. Use "
                "noise_schedule='linear' for ε-pred, or switch to pred_name='v'.",
                stacklevel=2,
            )

    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        pred_name=pred_name_map[pred_name],
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        snr_gamma=snr_gamma,
    )
