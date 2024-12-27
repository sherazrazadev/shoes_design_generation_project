from typing import List, Tuple, Union, Callable, Literal
import PIL
from tqdm import tqdm
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as T
from einops import rearrange, repeat
from einops_exts import check_shape
from .Unet import Unet
from .helpers import cast_tuple, default, resize_image_to, normalize_neg_one_to_one,
    unnormalize_zero_to_one, identity, exists, module_device, right_pad_dims_to, maybe, eval_decorator, null_context
from .t5 import t5_encode_text, get_encoded_dim
from .diffusion_model import GaussianDiffusion

class Victorai(nn.Module):
    def __init__(
            self,
            unets: Union[Unet, List[Unet], Tuple[Unet, ...]],
            *,
            text_encoder_name: str,
            image_sizes: Union[int, List[int], Tuple[int, ...]],
            text_embed_dim: int = None,
            channels: int = 3,
            timesteps: Union[int, List[int], Tuple[int, ...]] = 1000,
            cond_drop_prob: float = 0.1,
            loss_type: Literal["l1", "l2", "huber"] = 'l2',
            lowres_sample_noise_level: float = 0.2,
            auto_normalize_img: bool = True,
            dynamic_thresholding_percentile: float = 0.9,
            only_train_unet_number: int = None
    ):
        super().__init()
        self.loss_type = loss_type
        self.loss_fn = self._set_loss_fn(loss_type)
        self.channels = channels
        unets = cast_tuple(unets)
        num_unets = len(unets)
        self.noise_schedulers = self._make_noise_schedulers(num_unets, timesteps)
        self.lowres_noise_schedule = GaussianDiffusion(timesteps=timesteps)
        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))
        self.unet_being_trained_index = -1
        self.only_train_unet_number = only_train_unet_number
        self.unets = nn.ModuleList([])
        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, Unet)
            is_first = ind == 0
            one_unet = one_unet._cast_model_parameters(
                lowres_cond=not is_first,
                text_embed_dim=self.text_embed_dim,
                channels=self.channels,
                channels_out=self.channels,
            )
            self.unets.append(one_unet)
        self.image_sizes = cast_tuple(image_sizes)
        assert num_unets == len(image_sizes), f'you did not supply the correct number of u-nets ({len(self.unets)}) for resolutions {image_sizes}'
        self.sample_channels = cast_tuple(self.channels, num_unets)
        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.
        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile
        self.register_buffer('_temp', torch.tensor([0.]), persistent=False)
        self.to(next(self.unets.parameters()).device)

    @property
    def device(self) -> torch.device:
        return self._temp.device

    @staticmethod
    def _set_loss_fn(loss_type: str) -> Callable:
        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()
        return loss_fn

    @staticmethod
    def _make_noise_schedulers(
            num_unets: int,
            timesteps: Union[int, List[int], Tuple[int, ...]]
    ) -> Tuple[GaussianDiffusion, ...]:
        timesteps = cast_tuple(timesteps, num_unets)
        noise_schedulers = nn.ModuleList([])
        for timestep in timesteps:
            noise_scheduler = GaussianDiffusion(timesteps=timestep)
            noise_schedulers.append(noise_scheduler)
        return noise_schedulers

    def _get_unet(self, unet_number: int) -> Unet:
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1
        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list
        unet = self.unets[index]
        return unet

    def _get_noise_schedule(self, unet_number: int) -> GaussianDiffusion:
        assert 0 < unet_number <= len(self.noise_schedulers)
        return self.noise_schedulers[unet_number - 1]

    def forward(self, img, *, text=None, num_unet=None, model_kwargs=None, **kwargs):
        assert (num_unet is not None) or (text is not None), 'either supply num_unet, or text (or both)'
        is_first_unet = num_unet == 1
        is_last_unet = num_unet == len(self.unets)
        current_device = img.device
        assert num_unet is not None
        unet = self._get_unet(num_unet)
        assert img.shape[1] == self.channels
        if isinstance(self.unets, nn.ModuleList) and not is_first_unet:
            self.unets = nn.Sequential(*self.unets)
        else:
            self.unets = self.unets
        if not is_first_unet:
            with self.unnormalize():
                img = self.normalize_img(img)
        if text is not None:
            img = self._normalize_text_and_image(img, text)
        assert img.shape[2:] == self.image_sizes[num_unet - 1], f'unet {num_unet} expects image size {self.image_sizes[num_unet - 1]}, but got {img.shape[2:]}'
        if self.cond_drop_prob and not is_last_unet:
            unet = self._make_cond_dropout(unet)
        condition = None
        if num_unet < len(self.unets):
            with torch.no_grad():
                condition = self._get_condition(
                    unet_number=num_unet + 1,
                    current_image_size=img.shape[2:],
                    device=current_device
                )
        noise = self._get_noise(num_unet, img.shape, current_device)
        predicted_img = unet(img, noise, condition, **model_kwargs)
        return self._cleanup_output(predicted_img, img)

    @staticmethod
    def _normalize_text_and_image(img, text):
        img = img / 2 + 0.5
        text_embed = t5_encode_text(text)
        text_embed = F.normalize(text_embed, dim=-1)
        concat_dim = 2 if len(img.shape) == 4 else 1
        concatenated = torch.cat([img, text_embed.unsqueeze(concat_dim)], dim=concat_dim)
        concatenated = concatenated * 2 - 1
        return concatenated

    def _get_condition(self, unet_number, current_image_size, device):
        noise_schedule = self._get_noise_schedule(unet_number)
        current_resolution = current_image_size[0]
        noise = noise_schedule.sample_noise(
            (current_resolution, current_resolution),
            self.channels,
            device=device,
            sigma=self.lowres_sample_noise_level
        )
        return noise

    def _get_noise(self, num_unet, current_image_size, current_device):
        unet = self._get_unet(num_unet)
        noise = self._get_noise_schedule(num_unet).sample_noise(
            current_image_size, self.channels, device=current_device
        )
        noise = rearrange(noise, 'b c h w -> b c () h w')
        if unet.lowres_cond:
            cond_resolution = noise.shape[3]
            noise = F.interpolate(noise, size=(current_image_size[1], current_image_size[2]), mode='bilinear', align_corners=False)
            if noise.shape[3] != current_image_size[3]:
                noise = right_pad_dims_to(noise, (noise.shape[0], noise.shape[1], noise.shape[2], current_image_size[3]), pad_value=0)
            assert noise.shape[3] == current_image_size[3], f'noise resolution mismatch (expected {current_image_size[3]} but got {noise.shape[3]})'
            noise = noise.clamp(-1, 1)
        return noise

    def _make_cond_dropout(self, unet: Unet) -> nn.Module:
        if hasattr(unet, 'cond_drop_prob'):
            return unet
        cond_drop_prob = self.cond_drop_prob
        unet.cond_drop_prob = cond_drop_prob
        unet.cond_drop = nn.Dropout2d(cond_drop_prob)
        return unet

    def _cleanup_output(self, predicted_img, original_img):
        original_img = self.normalize_img(original_img)
        img_diff = predicted_img - original_img
        loss = self.loss_fn(predicted_img, original_img)
        img_range = self.input_image_range
        img_diff_norm = (img_diff - img_range[0]) / (img_range[1] - img_range[0])
        return {
            'prediction': self.unnormalize_img(predicted_img),
            'diff': img_diff_norm,
            'loss': loss
        }

    @contextmanager
    def unnormalize(self):
        original_norm = self.normalize_img
        self.normalize_img = identity
        self.unnormalize_img = identity
        yield
        self.normalize_img = original_norm
        self.unnormalize_img = unnormalize_zero_to_one

    def __repr__(self):
        unet_type = type(self.unets[0]).__name__
        return f'{unet_type} stack ({len(self.unets)} deep) {self.channels} channel(s), image sizes: {self.image_sizes}, unets output {self.channels} channel(s)'
