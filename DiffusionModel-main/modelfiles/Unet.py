from typing import Union

from einops import rearrange
from einops.layers.torch import Rearrange
from einops_exts.torch import EinopsToAndFrom
import torch
from torch import nn
import torch.nn.functional as F

from .helpers import default, exists, cast_tuple, prob_mask_like
from .layers import (
    Attention,
    CrossEmbedLayer,
    Downsample,
    Residual,
    ResnetBlock,
    SinusoidalPosEmb,
    TransformerBlock,
    Upsample, Parallel, Identity
)

from .t5 import get_encoded_dim


class Unet(nn.Module):
    

    def __init__(
            self,
            *,
            dim: int = 128,
            dim_mults: tuple = (1, 2, 4),
            channels: int = 3,
            channels_out: int = None,
            cond_dim: int = None,
            text_embed_dim=get_encoded_dim('t5_small'),
            num_resnet_blocks: Union[int, tuple] = 1,
            layer_attns: Union[bool, tuple] = True,
            layer_cross_attns: Union[bool, tuple] = True,
            attn_heads: int = 8,
            lowres_cond: bool = False,
            memory_efficient: bool = False,
            attend_at_middle: bool = False

    ):
       
        super().__init__()

        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)

        ATTN_DIM_HEAD = 64 
        NUM_TIME_TOKENS = 2 
        RESNET_GROUPS = 8 

        init_conv_to_final_conv_residual = False  
        final_resnet_block = True 

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1)

        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_cond_dim),
            nn.SiLU()
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * NUM_TIME_TOKENS),
            Rearrange('b (r d) -> b r d', r=NUM_TIME_TOKENS)
        )

        self.lowres_cond = lowres_cond
        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_cond_dim),
                nn.SiLU()
            )

            self.to_lowres_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )

            self.to_lowres_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * NUM_TIME_TOKENS),
                Rearrange('b (r d) -> b r d', r=NUM_TIME_TOKENS)
            )

        self.norm_cond = nn.LayerNorm(cond_dim)
        self.text_embed_dim = text_embed_dim
        self.text_to_cond = nn.Linear(self.text_embed_dim, cond_dim)

        max_text_len = 256
        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
        self.null_text_hidden = nn.Parameter(torch.randn(1, time_cond_dim))

        self.to_text_non_attn_cond = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, time_cond_dim),
            nn.SiLU(),
            nn.Linear(time_cond_dim, time_cond_dim)
        )
        self.channels = channels
        self.channels_out = default(channels_out, channels)
        self.init_conv = CrossEmbedLayer(channels if not lowres_cond else channels * 2,
                                         dim_out=dim,
                                         kernel_sizes=(3, 7, 15),
                                         stride=1)
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        num_resolutions = len(in_out)
        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_resolutions)
        resnet_groups = cast_tuple(RESNET_GROUPS, num_resolutions)
        layer_attns = cast_tuple(layer_attns, num_resolutions)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_resolutions)

        assert all(
            [layers == num_resolutions for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])

        self.skip_connect_scale = 2 ** -0.5

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_cross_attns]
        reversed_layer_params = list(map(reversed, layer_params))

        skip_connect_dims = []
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_cross_attn) in enumerate(
                zip(in_out, *layer_params)):

            is_last = ind == (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn else None
            transformer_block_klass = TransformerBlock if layer_attn else Identity

            current_dim = dim_in
            pre_downsample = None
            if memory_efficient:
                pre_downsample = Downsample(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)
            post_downsample = None
            if not memory_efficient:
                post_downsample = Downsample(current_dim, dim_out) if not is_last else Parallel(
                    nn.Conv2d(dim_in, dim_out, 3, padding=1), nn.Conv2d(dim_in, dim_out, 1))
            self.downs.append(nn.ModuleList([
                pre_downsample,
                ResnetBlock(current_dim,
                            current_dim,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups),
                nn.ModuleList(
                    [
                        ResnetBlock(current_dim,
                                    current_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups
                                    )
                        for _ in range(layer_num_resnet_blocks)
                    ]
                ),
                transformer_block_klass(dim=current_dim,
                                        heads=attn_heads,
                                        dim_head=ATTN_DIM_HEAD),
                post_downsample,
            ]))
        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=time_cond_dim,
                                      groups=resnet_groups[-1])
        self.mid_attn = EinopsToAndFrom('b c h w', 'b (h w) c',
                                        Residual(Attention(mid_dim, heads=attn_heads,
                                                           dim_head=ATTN_DIM_HEAD))) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=time_cond_dim,
                                      groups=resnet_groups[-1])

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_cross_attn) in enumerate(
                zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (num_resolutions - 1)
            layer_cond_dim = cond_dim if layer_cross_attn else None
            transformer_block_klass = TransformerBlock if layer_attn else Identity

            skip_connect_dim = skip_connect_dims.pop()

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + skip_connect_dim,
                            dim_out,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups),
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out + skip_connect_dim,
                                    dim_out,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups)
                        for _ in range(layer_num_resnet_blocks)
                    ]),
                transformer_block_klass(dim=dim_out,
                                        heads=attn_heads,
                                        dim_head=ATTN_DIM_HEAD),
                Upsample(dim_out, dim_in) if not is_last or memory_efficient else Identity()
            ]))
        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = dim * (2 if init_conv_to_final_conv_residual else 1)

        self.final_res_block = ResnetBlock(final_conv_dim, dim, time_cond_dim=time_cond_dim,
                                           groups=resnet_groups[0]) if final_resnet_block else None
        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        self.final_conv = nn.Conv2d(final_conv_dim_in, self.channels_out, 3,
                                    padding=3 // 2)
    def _cast_model_parameters(
            self,
            *,
            lowres_cond,
            text_embed_dim,
            channels,
            channels_out,
    ):
        if lowres_cond == self.lowres_cond and \
                channels == self.channels and \
                text_embed_dim == self.text_embed_dim and \
                channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond=lowres_cond,
            text_embed_dim=text_embed_dim,
            channels=channels,
            channels_out=channels_out,
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    def forward(
            self,
            x: torch.tensor,
            time: torch.tensor,
            *,
            lowres_cond_img: torch.tensor = None,
            lowres_noise_times: torch.tensor = None,
            text_embeds: torch.tensor = None,
            text_mask: torch.tensor = None,
            cond_drop_prob: float = 0.
            ) -> torch.tensor:

        batch_size, device = x.shape[0], x.device

        assert not (self.lowres_cond and not exists(lowres_cond_img)), \
            'low resolution conditioning image must be present'
        assert not (self.lowres_cond and not exists(lowres_noise_times)), \
            'low resolution conditioning noise time must be present'

        t, time_tokens = self._generate_t_tokens(time, lowres_noise_times)

        t, c = self._text_condition(text_embeds, batch_size, cond_drop_prob, device, text_mask, t, time_tokens)

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim=1)
        x = self.init_conv(x)
        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()
        hiddens = []
        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:

            if exists(pre_downsample):
                x = pre_downsample(x)
            x = init_block(x, t, c)
            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)
            x = attn_block(x)
            hiddens.append(x)
            if exists(post_downsample):
                x = post_downsample(x)
        x = self.mid_block1(x, t, c)
        if exists(self.mid_attn):
            x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)
        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim=1)

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = add_skip_connection(x)
            x = init_block(x, t, c)
            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)
            x = attn_block(x)
            x = upsample(x)
        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim=1)
        if exists(self.final_res_block):
            x = self.final_res_block(x, t)
        return self.final_conv(x)

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale: float = 1.,
            **kwargs
    ) -> torch.tensor:
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def _generate_t_tokens(
            self,
            time: torch.tensor,
            lowres_noise_times: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        time_hiddens = self.to_time_hiddens(time)
        t = self.to_time_cond(time_hiddens)
        time_tokens = self.to_time_tokens(time_hiddens)
        if self.lowres_cond:
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            lowres_time_tokens = self.to_lowres_time_tokens(lowres_time_hiddens)
            lowres_t = self.to_lowres_time_cond(lowres_time_hiddens)

            t = t + lowres_t
            time_tokens = torch.cat((time_tokens, lowres_time_tokens), dim=-2)

        return t, time_tokens

    def _text_condition(
            self,
            text_embeds: torch.tensor,
            batch_size: int,
            cond_drop_prob: float,
            device: torch.device,
            text_mask: torch.tensor,
            t: torch.tensor,
            time_tokens: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        text_tokens = None
        if exists(text_embeds):
            text_tokens = self.text_to_cond(text_embeds)
            text_tokens = text_tokens[:, :self.max_text_len]
            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len
            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))
            text_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')
            if exists(text_mask):
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value=False)

                text_mask = rearrange(text_mask, 'b n -> b n 1') 
                text_keep_mask_embed = text_mask & text_keep_mask_embed 
            null_text_embed = self.null_text_embed.to(text_tokens.dtype) 
            text_tokens = torch.where(
                text_keep_mask_embed,
                text_tokens,
                null_text_embed
            )

            mean_pooled_text_tokens = text_tokens.mean(dim=-2)
            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)
            null_text_hidden = self.null_text_hidden.to(t.dtype)
            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')
            text_hiddens = torch.where(
                text_keep_mask_hidden,
                text_hiddens,
                null_text_hidden
            )

            t = t + text_hiddens

        c = time_tokens if not exists(text_tokens) else torch.cat((time_tokens, text_tokens), dim=-2)

        c = self.norm_cond(c)

        return t, c


class Base(Unet):
    defaults = dict(
        dim=512,
        dim_mults=(1, 2, 3, 4),
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True),
        memory_efficient=False
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**Base.defaults, **kwargs})


class Super(Unet):
    defaults = dict(
        dim=128,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=(2, 4, 8, 8),
        layer_attns=(False, False, False, True),
        layer_cross_attns=(False, False, False, True),
        memory_efficient=True
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**Super.defaults, **kwargs})


class BaseTest(Unet):
    defaults = dict(
        dim=8,
        dim_mults=(1, 2),
        num_resnet_blocks=1,
        layer_attns=False,
        layer_cross_attns=False,
        memory_efficient=False
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**Base.defaults, **kwargs})


class SuperTest(Unet):
    defaults = dict(
        dim=8,
        dim_mults=(1, 2),
        num_resnet_blocks=(1, 2),
        layer_attns=False,
        layer_cross_attns=False,
        memory_efficient=True
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**Super.defaults, **kwargs})
