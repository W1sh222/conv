"""Small CPU-testable losses for Top-p continuation and anchor replay."""
import torch
import torch.nn.functional as F


def row_distribution(energy, rows, temperature=0.015, positive_mass=False):
    """Normalize real causal blocks only, never future or estimator padding."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    idx = torch.as_tensor(rows, device=energy.device, dtype=torch.long)
    values = energy.index_select(2, idx).float()
    valid = torch.arange(energy.shape[-1], device=energy.device)[None, :] <= idx[:, None]
    valid = valid[None, None]
    mass = values.clamp_min(0) if positive_mass else F.softplus(values / temperature) * temperature
    mass = mass.masked_fill(~valid, 0)
    total = mass.sum(-1, keepdim=True)
    # Inference keeps all causal blocks on zero-mass rows. Uniform is a
    # finite training distribution for that case; it is not attention recall.
    fallback = valid.float() / valid.float().sum(-1, keepdim=True).clamp_min(1)
    return torch.where(total > 0, mass / total.clamp_min(1e-8), fallback)


def distribution_kl(predicted, target):
    # The exact teacher is built under inference_mode. Clone so autograd may
    # save this target when differentiating the prediction.
    target = target.detach().float().clone()
    return (target * (target.clamp_min(1e-8).log() - predicted.clamp_min(1e-8).log())).sum(-1).mean()


def anchor_replay_loss(energy, anchor_energy, rows, temperature):
    """Retain the starting selector's soft AND corrected positive distributions."""
    return sum(distribution_kl(
        row_distribution(energy, rows, temperature, positive),
        row_distribution(anchor_energy.detach(), rows, temperature, positive))
        for positive in (False, True)) / 2
