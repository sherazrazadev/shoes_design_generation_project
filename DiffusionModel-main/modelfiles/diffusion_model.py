import torch
import torch.nn.functional as F
from torch import nn
from .helpers import default, log, extract

class GaussianDiffusion(nn.Module):

    def __init__(self, timesteps: int):
        super().__init__()

        assert not timesteps < 20, 'timesteps must be at least 20'
        self.num_timesteps = timesteps

        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent=False)

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', log(posterior_variance, eps=1e-20))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def _get_times(self, batch_size: int, noise_level: float, device: torch.device) -> torch.tensor:
        return torch.full((batch_size,), int(self.num_timesteps * noise_level), device=device, dtype=torch.long)

    def _sample_random_times(self, batch_size: int, device: torch.device) -> torch.tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

    def _get_sampling_timesteps(self, batch: int, device: torch.device) -> list[torch.tensor]:
        time_transitions = []
        for i in reversed(range(self.num_timesteps)):
            time_transitions.append((torch.full((batch,), i, device=device, dtype=torch.long)))
        return time_transitions

    def q_posterior(self, x_start: torch.tensor, x_t: torch.tensor, t: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start: torch.tensor, t: torch.tensor, noise: torch.tensor = None) -> torch.tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        noised = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return noised

    def predict_start_from_noise(self, x_t: torch.tensor, t: torch.tensor, noise: torch.tensor) -> torch.tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
