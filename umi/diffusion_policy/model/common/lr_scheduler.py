"""
Dependency-free learning-rate schedulers.

Local replacement for the `diffusers.optimization.get_scheduler` family that
diffusion_policy's workspaces use (cosine / linear / constant, with warmup).
Same semantics as the diffusers versions (stepped every batch, closures over
a step counter). Removes the diffusers dependency (plan C, 2026-08-26).
"""
import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_constant_lambda(_=None):
    return 1


def _get_constant_schedule(optimizer, last_epoch=-1):
    return LambdaLR(optimizer, _get_constant_lambda, last_epoch=last_epoch)


def _get_constant_schedule_with_warmup_lr_lambda(current_step, *, num_warmup_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def _get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    lr_lambda = partial(
        _get_constant_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_linear_schedule_with_warmup_lr_lambda(
        current_step, *, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)))


def _get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step, *, num_warmup_steps, num_training_steps, num_cycles):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps))
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def _get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps,
        num_cycles=0.5, last_epoch=-1):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


_TYPE_TO_SCHEDULER_FUNCTION = {
    'constant': _get_constant_schedule,
    'constant_with_warmup': _get_constant_schedule_with_warmup,
    'linear': _get_linear_schedule_with_warmup,
    'cosine': _get_cosine_schedule_with_warmup,
    'cosine_with_restarts': _get_cosine_schedule_with_warmup,
}


def get_scheduler(
    name,
    optimizer: Optimizer,
    num_warmup_steps=None,
    num_training_steps=None,
    **kwargs
):
    """Unified API to get any scheduler from its name (diffusers semantics).

    Args:
        name: 'constant' | 'constant_with_warmup' | 'linear' | 'cosine' |
            'cosine_with_restarts'
        optimizer: torch optimizer the scheduler operates on.
        num_warmup_steps: warmup steps (required except for 'constant').
        num_training_steps: total training steps (required except for the
            constant schedules).
        **kwargs: forwarded, e.g. last_epoch=..., num_cycles=...
    """
    name = str(name)
    if name not in _TYPE_TO_SCHEDULER_FUNCTION:
        raise ValueError(f'{name} is not a valid scheduler name')

    schedule_func = _TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == 'constant':
        return schedule_func(optimizer, **kwargs)

    if num_warmup_steps is None:
        raise ValueError(f'{name} requires `num_warmup_steps`')

    if name == 'constant_with_warmup':
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    if num_training_steps is None:
        raise ValueError(f'{name} requires `num_training_steps`')

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs)
