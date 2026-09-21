"""
Dependency-free DDPM noise scheduler.

Faithful port of the subset of `diffusers.schedulers.scheduling_ddpm.DDPMScheduler`
(0.18.x) used by diffusion_policy: linear / scaled_linear / squaredcos_cap_v2
beta schedules, add_noise, epsilon/v_prediction/sample denoising step with
clip_sample and fixed variance types, and set_timesteps for inference.

Replaces the diffusers + huggingface_hub dependency entirely (plan C,
2026-08-26). Only the config fields the pipeline touches are modeled.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor


class DDPMConfig:
    """Minimal attribute bag matching the diffusers config fields we use."""

    def __init__(self, **kwargs):
        self.num_train_timesteps = int(kwargs.get('num_train_timesteps', 1000))
        self.beta_start = float(kwargs.get('beta_start', 0.0001))
        self.beta_end = float(kwargs.get('beta_end', 0.02))
        self.beta_schedule = kwargs.get('beta_schedule', 'linear')
        self.trained_betas = kwargs.get('trained_betas', None)
        self.variance_type = kwargs.get('variance_type', 'fixed_small')
        self.clip_sample = bool(kwargs.get('clip_sample', True))
        self.clip_sample_range = float(kwargs.get('clip_sample_range', 1.0))
        self.prediction_type = kwargs.get('prediction_type', 'epsilon')
        self.thresholding = bool(kwargs.get('thresholding', False))
        self.dynamic_thresholding_ratio = float(
            kwargs.get('dynamic_thresholding_ratio', 0.995))
        self.sample_max_value = float(kwargs.get('sample_max_value', 1.0))
        self.steps_offset = int(kwargs.get('steps_offset', 0))
        self.rescale_betas_zero_snr = bool(
            kwargs.get('rescale_betas_zero_snr', False))
        self.timestep_spacing = kwargs.get('timestep_spacing', 'leading')


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """beta schedule for squaredcos_cap_v2 (diffusers formula)."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


@dataclass
class DDPMOutput:
    prev_sample: Tensor


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        trained_betas: Optional[np.ndarray] = None,
        variance_type: str = 'fixed_small',
        clip_sample: bool = True,
        prediction_type: str = 'epsilon',
        **kwargs
    ):
        self.config = DDPMConfig(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            variance_type=variance_type,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
            **kwargs)

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == 'linear':
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == 'scaled_linear':
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5,
                num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == 'squaredcos_cap_v2':
            self.betas = betas_for_alpha_bar(
                num_train_timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
        else:
            raise NotImplementedError(
                f'{beta_schedule} does is not implemented for {self.__class__}')

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0
        self.timesteps = None
        self.num_inference_steps = None

        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        # required for add_noise
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # required for reconstruct x0 from x_t
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1.0 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod))

        # posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod))

        # log computation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20))

        if variance_type == 'fixed_large':
            self.posterior_log_variance_clipped = torch.log(
                torch.cat([self.posterior_variance[1:2], self.betas[1:]]))
        elif variance_type not in (
                'fixed_small', 'fixed_small_log', 'fixed_large'):
            raise NotImplementedError(
                f'variance_type {variance_type} not supported')

    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        prev_t = self._prev_timestep(t)

        alpha_prod_t = self.alphas_cumprod.to(t.device)[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod.to(t.device)[prev_t]
            if prev_t >= 0 else self.one.to(t.device))
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance beta_t
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        # we always take the log of variance, so clamp it to ensure it's never 0
        variance = torch.clamp(variance, min=1e-20)

        if variance_type is None:
            variance_type = self.config.variance_type

        if variance_type == 'fixed_small':
            return variance
        elif variance_type == 'fixed_small_log':
            return torch.log(variance)
        elif variance_type == 'fixed_large':
            return torch.log(torch.max(variance, torch.tensor(0.1, device=variance.device)))
        raise NotImplementedError(variance_type)

    def _prev_timestep(self, timestep):
        num_inference_steps = (
            self.num_inference_steps
            if self.num_inference_steps else self.config.num_train_timesteps)
        prev_t = timestep - self.config.num_train_timesteps // num_inference_steps
        return prev_t

    def set_timesteps(self, num_inference_steps: int, device=None):
        """DDPM inference timesteps: evenly spaced, reversed, int64."""
        self.num_inference_steps = num_inference_steps
        timesteps = (
            np.linspace(
                0, self.config.num_train_timesteps - 1,
                num_inference_steps)
            .round()[::-1].copy().astype(np.int64))
        self.timesteps = torch.from_numpy(timesteps)
        if device is not None:
            self.timesteps = self.timesteps.to(device)

    def add_noise(self, original_samples, noise, timesteps):
        """Forward diffusion q(x_t | x_0): x_t = sqrt(a_cum) x_0 + sqrt(1-a_cum) eps."""
        sqrt_alpha_prod = self.alphas_cumprod.to(original_samples.device)[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (
            (1 - self.alphas_cumprod).to(original_samples.device)[timesteps] ** 0.5)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples
            + sqrt_one_minus_alpha_prod * noise)
        return noisy_samples

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        generator=None,
        return_dict: bool = True,
        **kwargs
    ):
        """Reverse diffusion step (epsilon / v_prediction / sample)."""
        t = timestep

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod.to(sample.device)[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod.to(sample.device)[self._prev_timestep(t)]
            if self._prev_timestep(t) >= 0
            else self.one.to(sample.device))
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise
        if self.config.prediction_type == 'epsilon':
            pred_original_sample = (
                (sample - beta_prod_t ** (0.5) * model_output)
                / alpha_prod_t ** (0.5))
        elif self.config.prediction_type == 'sample':
            pred_original_sample = model_output
        elif self.config.prediction_type == 'v_prediction':
            pred_original_sample = (
                (alpha_prod_t ** 0.5) * sample
                - (beta_prod_t ** 0.5) * model_output)
        else:
            raise ValueError(
                f'prediction_type {self.config.prediction_type} not supported')

        # 3. clip or threshold "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample,
                -self.config.clip_sample_range, self.config.clip_sample_range)

        # 4. compute coefficients for pred_original_sample x_0 and current x_t
        pred_original_sample_coeff = (
            (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t)
        current_sample_coeff = (
            current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t)

        # 5. compute predicted previous sample mean
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample)

        # 6. add noise
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape, generator=generator,
                device=device, dtype=model_output.dtype)
            variance = self._get_variance(t).to(device) ** (0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMOutput(prev_sample=pred_prev_sample)

    def get_velocity(self, sample, noise, timesteps):
        """v-prediction velocity (unused by current configs, kept for completeness)."""
        sqrt_alpha_prod = self.alphas_cumprod.to(sample.device)[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (
            (1 - self.alphas_cumprod).to(sample.device)[timesteps] ** 0.5)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


class DDIMScheduler(DDPMScheduler):
    """Dependency-free DDIM scheduler (eta=0 deterministic by default).

    Port of the subset of `diffusers.schedulers.scheduling_ddim.DDIMScheduler`
    (0.18.x) used by diffusion_policy's transformer policies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DDIM-specific config defaults
        self.config.prediction_type = kwargs.get('prediction_type', 'epsilon')
        self.config.clip_sample = kwargs.get('clip_sample', False)
        self.config.set_alpha_to_one = kwargs.get('set_alpha_to_one', True)

    def _prev_timestep(self, timestep):
        num_inference_steps = (
            self.num_inference_steps
            if self.num_inference_steps else self.config.num_train_timesteps)
        return timestep - self.config.num_train_timesteps // num_inference_steps

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise=None,
        return_dict: bool = True,
        **kwargs
    ):
        """Reverse DDIM step (epsilon prediction, optional stochasticity)."""
        t = timestep

        # 1. get previous step value
        prev_t = self._prev_timestep(t)

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod.to(sample.device)[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod.to(sample.device)[prev_t]
            if prev_t >= 0 else self.one.to(sample.device))
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample
        if self.config.prediction_type == 'epsilon':
            if use_clipped_model_output:
                # model_output is the predicted original sample x0 here
                pred_original_sample = model_output
                pred_epsilon = (
                    (sample - alpha_prod_t ** (0.5) * pred_original_sample)
                    / beta_prod_t ** (0.5))
            else:
                pred_original_sample = (
                    (sample - beta_prod_t ** (0.5) * model_output)
                    / alpha_prod_t ** (0.5))
                pred_epsilon = model_output
        elif self.config.prediction_type == 'sample':
            pred_original_sample = model_output
            pred_epsilon = (
                (sample - alpha_prod_t ** (0.5) * pred_original_sample)
                / beta_prod_t ** (0.5))
        elif self.config.prediction_type == 'v_prediction':
            pred_original_sample = (
                (alpha_prod_t ** 0.5) * sample
                - (beta_prod_t ** 0.5) * model_output)
            pred_epsilon = (
                (alpha_prod_t ** 0.5) * model_output
                + (beta_prod_t ** 0.5) * sample)
        else:
            raise ValueError(
                f'prediction_type {self.config.prediction_type} not supported')

        # 4. clip or threshold
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample,
                -self.config.clip_sample_range, self.config.clip_sample_range)

        # 5. compute variance: "sigma_t(eta)"
        alpha_prod_t_prev_ = alpha_prod_t_prev
        variance = (beta_prod_t_prev := 1 - alpha_prod_t_prev_) / beta_prod_t * (1 - alpha_prod_t / alpha_prod_t_prev_)
        std_dev_t = eta * variance ** (0.5)

        # 6. compute "direction pointing to x_t"
        pred_sample_direction = (
            (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon)

        # 7. compute x_t without "random noise"
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample
            + pred_sample_direction)

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    'Cannot pass both generator and variance_noise.')
            if variance_noise is None:
                variance_noise = torch.randn(
                    model_output.shape, generator=generator,
                    device=model_output.device, dtype=model_output.dtype)
            variance = std_dev_t * variance_noise
            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample,)

        return DDIMOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


@dataclass
class DDIMOutput:
    prev_sample: Tensor
    pred_original_sample: Optional[Tensor] = None
