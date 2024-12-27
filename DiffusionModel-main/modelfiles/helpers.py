from contextlib import contextmanager
from functools import wraps
import torch
from typing import Literal, Callable
from resize_right import resize

def cast_tuple(val, length: int = None) -> tuple:
    if isinstance(val, list):
        val = tuple(val)
    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))
    if exists(length):
        assert len(output) == length
    return output

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def exists(val) -> bool:
    return val is not None

def extract(a: torch.tensor, t: torch.tensor, x_shape: torch.Size) -> torch.tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1))

def identity(t, *args, **kwargs):
    return t

def log(t: torch.tensor, eps: float = 1e-12) -> torch.tensor:
    return torch.log(t.clamp(min=eps)

def maybe(fn: Callable) -> Callable:
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner

def module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device

def normalize_neg_one_to_one(img: torch.tensor) -> torch.tensor:
    return img * 2 - 1

@contextmanager
def null_context(*args, **kwargs):
    yield

def prob_mask_like(shape: tuple, prob: float, device: torch.device) -> torch.Tensor:
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

def resize_image_to(image: torch.tensor, target_image_size: int, clamp_range: tuple = None, pad_mode: Literal['constant', 'edge', 'reflect', 'symmetric'] = 'reflect') -> torch.tensor:
    orig_image_size = image.shape[-1]
    if orig_image_size == target_image_size:
        return image
    scale_factors = target_image_size / orig_image_size
    out = resize(image, scale_factors=scale_factors, pad_mode=pad_mode)
    if exists(clamp_range):
        out = out.clamp(*clamp_range)
    return out

def right_pad_dims_to(x: torch.tensor, t: torch.tensor) -> torch.tensor:
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5
