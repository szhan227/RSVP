import sys
from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchsummary import summary
from omegaconf import OmegaConf
import utils

from model.ldm.diffusionmodules import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

logger = utils.logger
class DiffusionWrapper(nn.Module):
    def __init__(self, model, conditioning_key=None):
        super().__init__()
        self.diffusion_model = model
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, cond, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, cond, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class AttentionBlock1D(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, l = x.shape
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    # @staticmethod
    # def count_flops(model, _x, y):
    #     return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    # @staticmethod
    # def count_flops(model, _x, y):
    #     return count_flops_attn(model, _x, y)

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        # self.norm = Normalize(in_channels)
        self.norm = nn.LayerNorm(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            # TODO
            # [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
            #     for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        ds_bg,
        ds_id,
        ds_mo,
        vae_hidden,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        cond_model=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.cond_model = cond_model
        # self.num_frames = num_frames
        if cond_model:
            self.register_buffer("zeros", torch.zeros(1, self.in_channels, 2048))

        self.ds_bg = ds_bg
        self.ds_id = ds_id
        self.ds_mo = ds_mo
        self.vae_hidden = vae_hidden

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # TODO: try to change in_channels to num_frames
        if cond_model:
            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(dims, in_channels*2, model_channels, 3, padding=1)
                        # conv_nd(dims, num_frames*2, model_channels, 3, padding=1)
                    )
                ]
            )
        else:
            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(dims, in_channels, model_channels, 3, padding=1)
                        # conv_nd(dims, num_frames, model_channels, 3, padding=1)
                    )
                ]
            )

        # # TODO: input_static_blocks, the convolutional layers that handle background and object latents
        # self.input_static_blocks = nn.ModuleList(
        #     [
        #         TimestepEmbedSequential(
        #             conv_nd(dims, 2 if cond_model else 1, model_channels, 3, padding=1)
        #         )
        #     ]
        # )
        # # TODO END

        self.input_attns = nn.ModuleList([nn.Identity()])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # in our case, mult: 1, 2, 4, 4
        logger.debug('before loop:')
        logger.debug('num of input blocks:', len(self.input_blocks))
        logger.debug('num of input attns:', len(self.input_attns))
        for level, mult in enumerate(channel_mult):
            # logger.debug('in level:', level, 'mult:', mult)
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                # logger.debug('model channels:', ch)
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                # self.input_static_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

                self.input_attns.append(
                            AttentionBlock1D(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ))


            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                #
                # self.input_static_blocks.append(
                #     TimestepEmbedSequential(
                #         ResBlock(
                #             ch,
                #             time_embed_dim,
                #             dropout,
                #             out_channels=out_ch,
                #             dims=dims,
                #             use_checkpoint=use_checkpoint,
                #             use_scale_shift_norm=use_scale_shift_norm,
                #             down=True,
                #         )
                #         if resblock_updown
                #         else Downsample(
                #             ch, conv_resample, dims=dims, out_channels=out_ch
                #         )
                #     )
                # )
                #
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

                self.input_attns.append(
                            AttentionBlock1D(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ))
            # logger.debug('num of input blocks:', len(self.input_blocks))
            # logger.debug('num of input attns:', len(self.input_attns))

        # logger.debug('---------------------------------------')
        # print(self.input_blocks)
        # logger.debug('---------------------------------------')
        # print(self.input_attns)
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.mid_attn = AttentionBlock1D(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        self.output_attns = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_attns.append(AttentionBlock1D(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    # def convert_to_fp16(self):
    #     """
    #     Convert the torso of the model to float16.
    #     """
    #     self.input_blocks.apply(convert_module_to_f16)
    #     self.middle_block.apply(convert_module_to_f16)
    #     self.output_blocks.apply(convert_module_to_f16)
    #
    # def convert_to_fp32(self):
    #     """
    #     Convert the torso of the model to float32.
    #     """
    #     self.input_blocks.apply(convert_module_to_f32)
    #     self.middle_block.apply(convert_module_to_f32)
    #     self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, cond=None, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        logger.info('show timesteps: ', timesteps, timesteps.shape)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        h_bgs = []
        h_mos = []
        h_ids = []


        # if timesteps is None:
        #     timesteps = torch.tensor([0, 1, 2, 3, 4]).to(x.device)
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        
        emb = self.time_embed(t_emb)


        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        if cond != None:
            h = torch.cat([h, cond], dim=1)
            logger.debug('here concat z and h together in time dimension')
        elif self.cond_model:

            h = torch.cat([h, self.zeros.repeat(h.size(0), 1, 1)], dim=1)

        logger.debug('h shape: ', h.shape)
        # logger.debug('zeros shape: ', self.zeros.repeat(h.size(0), 1, 1).shape)
        # TODO: treat 32 and 16 as variables

        # h_bg = h[:, :, 0:32*32].view(h.size(0), h.size(1), 32, 32)
        # h_id = h[:, :, 32*32:32*(32+16)].view(h.size(0), h.size(1), 16, 32)
        # h_mo = h[:, :, 32*(32+16):32*(32+16+16)].view(h.size(0), h.size(1), 16, 32)
        bg_len = self.vae_hidden // 2 ** self.ds_bg  # 256 // 8 = 32
        id_len = self.vae_hidden // 2 ** self.ds_id  # 256 // 16 = 16
        mo_len = int(math.sqrt(h.size(-1) - bg_len ** 2 - id_len ** 2))  # 32
        # h_bg = h[:, :, :32 * 32].view(h.size(0), h.size(1), 32, 32)
        # h_id = h[:, :, 32 * 32: 32 * 32 + 16 * 16].view(h.size(0), h.size(1), 16, 16)
        # h_mo = h[:, :, 32 * 32 + 16 * 16:32 * 32 + 16 * 16 + 32 * 32].view(h.size(0), h.size(1), 32, 32)

        h_bg, h_id, h_mo = h.split([bg_len ** 2, id_len ** 2, mo_len ** 2], dim=2)
        h_bg = h_bg.view(h.size(0), h.size(1), bg_len, bg_len)
        h_id = h_id.view(h.size(0), h.size(1), id_len, id_len)
        h_mo = h_mo.view(h.size(0), h.size(1), mo_len, mo_len)
        # h_bg, h_id, h_mo = torch.chunk(h, 3, dim=-1)

        # trial: reconstruct hc and hx based on only 1 frame for bg and id
        # hc, hx = torch.chunk(h, 2, dim=1) # split by
        # hx_bg, hx_id = hx[:, :1, :, :], hx[:, 1:2, :, :]
        # hx_mo = hx[:, 2:, :, :]
        # hc_bg, hc_id = hc[:, :1, :, :], hc[:, 1:2, :, :]
        # hc_mo = hc[:, 2:, :, :]

        # concat the two parts in time dimension
        # h_bg = torch.cat([hc_bg, hx_bg], dim=1)
        # h_id = torch.cat([hc_id, hx_id], dim=1)
        # h_mo = torch.cat([hc_mo, hx_mo], dim=1)
        #___________________________________________________________


        logger.debug('h_bg shape: ', h_bg.shape)
        logger.debug('h_id shape: ', h_id.shape)
        logger.debug('h_mo shape: ', h_mo.shape)
        # logger.debug('input_blocks: ', len(self.input_blocks))
        # logger.debug('input_attns : ', len(self.input_attns))
        # logger.debug('start unet down sampling')
        counter = 0
        for module, input_attn in zip(self.input_blocks, self.input_attns):
            # logger.debug('\tinput_attn: ', input_attn)
            h_bg = module(h_bg, emb, context)
            h_id = module(h_id, emb, context)
            h_mo = module(h_mo, emb, context)

            logger.debug('\th_bg shape after module: ', h_bg.shape)
            logger.debug('\th_id shape after module: ', h_id.shape)
            logger.debug('\th_mo shape after module: ', h_mo.shape)

            # TODO: try to change res and t
            res = h_bg.size(-2)
            t   = h_id.size(-2)
            logger.debug('\tres, t: ', res, t)

            h_bg = h_bg.view(h_bg.size(0), h_bg.size(1), -1)
            h_id = h_id.view(h_id.size(0), h_id.size(1), -1)
            h_mo = h_mo.view(h_mo.size(0), h_mo.size(1), -1)

            logger.debug('\th_bg shape after view: ', h_bg.shape)
            logger.debug('\th_id shape after view: ', h_id.shape)
            logger.debug('\th_mo shape after view: ', h_mo.shape)

            # concatenate the three parts in last dimension
            h = torch.cat([h_bg, h_id, h_mo], dim=-1)
            h = input_attn(h)
            input_channels = 4
            logger.debug('\th shape after input_attn: ', h.shape)
            # h = h.reshape(h.size(0), h.size(1), input_channels, -1)
            # res = 32
            # t = 16
            h_bg = h[:, :, :res*res].view(h.size(0), h.size(1), res, res)
            h_id = h[:, :, res*res:res*res + t*t].view(h.size(0), h.size(1), t, t)
            h_mo = h[:, :, res*res+t*t:res*res+t*t + res * res].view(h.size(0), h.size(1), res, res)
            # h_bg, h_id, h_mo = torch.chunk(h, 3, dim=-1)

            logger.debug('\th_bg shape after res*res: ', h_bg.shape)
            logger.debug('\th_id shape after res*res: ', h_id.shape)
            logger.debug('\th_mo shape after res*res: ', h_mo.shape)

            h_bgs.append(h_bg)
            h_ids.append(h_id)
            h_mos.append(h_mo)

            logger.debug('\tin loop:', counter)
            logger.debug('\tappend h_bg shape: ', h_bg.shape)
            logger.debug('\tappend h_id shape: ', h_id.shape)
            logger.debug('\tappend h_mo shape: ', h_mo.shape)
            logger.debug()
            counter += 1

        h_bg = self.middle_block(h_bg, emb, context)
        h_id = self.middle_block(h_id, emb, context)
        h_mo = self.middle_block(h_mo, emb, context)

        res = h_bg.size(-2)
        t   = h_id.size(-2)

        h_bg = h_bg.view(h_bg.size(0), h_bg.size(1), -1)
        h_id = h_id.view(h_id.size(0), h_id.size(1), -1)
        h_mo = h_mo.view(h_mo.size(0), h_mo.size(1), -1)

        h = torch.cat([h_bg, h_id, h_mo], dim=-1)
        h = self.mid_attn(h)

        # h_bg = h[:, :, :res * res].view(h.size(0), h.size(1), res, res)
        # h_id = h[:, :, res * res:res * res + t * t].view(h.size(0), h.size(1), t, t)
        # h_mo = h[:, :, res * res + t * t:res * res + t * t + res * res].view(h.size(0), h.size(1), res, res)

        h_bg, h_id, h_mo = h.split([res ** 2, t ** 2, res ** 2], dim=2)
        h_bg = h_bg.view(h.size(0), h.size(1), res, res)
        h_id = h_id.view(h.size(0), h.size(1), t, t)
        h_mo = h_mo.view(h.size(0), h.size(1), res, res)

        logger.debug('start up sampling')
        for module, output_attn in zip(self.output_blocks, self.output_attns):
            # logger.debug('\tmodule: ', module)
            # logger.debug('\toutput_attn: ', output_attn)
            h_bg = th.cat([h_bg, h_bgs.pop()], dim=1)
            h_bg = module(h_bg, emb, context)
            h_id = th.cat([h_id, h_ids.pop()], dim=1)
            h_id = module(h_id, emb, context)
            h_mo = th.cat([h_mo, h_mos.pop()], dim=1)
            h_mo = module(h_mo, emb, context)

            res = h_bg.size(-2)
            t   = h_id.size(-2)

            h_bg = h_bg.view(h_bg.size(0), h_bg.size(1), -1)
            h_id = h_id.view(h_id.size(0), h_id.size(1), -1)
            h_mo = h_mo.view(h_mo.size(0), h_mo.size(1), -1)

            h = torch.cat([h_bg, h_id, h_mo], dim=-1)
            h = output_attn(h)

            # h_bg = h[:, :, :res*res].view(h.size(0), h.size(1), res, res)
            # h_id = h[:, :, res*res:res*res + t*t].view(h.size(0), h.size(1), t, t)
            # h_mo = h[:, :, res*res+t*t:res*res+t*t + res * res].view(h.size(0), h.size(1), res, res)
            h_bg, h_id, h_mo = h.split([res ** 2, t ** 2, res ** 2], dim=2)
            h_bg = h_bg.view(h.size(0), h.size(1), res, res)
            h_id = h_id.view(h.size(0), h.size(1), t, t)
            h_mo = h_mo.view(h.size(0), h.size(1), res, res)
            # logger.debug('\tappend h_bg shape: ', h_bg.shape)
            # logger.debug('\tappend h_id shape: ', h_id.shape)
            # logger.debug('\tappend h_mo shape: ', h_mo.shape)
            # print()

        h_bg = self.out(h_bg)
        h_id = self.out(h_id)
        h_mo = self.out(h_mo)
        # logger.debug('out h_bg shape: ', h_bg.shape)
        # logger.debug('out h_id shape: ', h_id.shape)
        # logger.debug('out h_mo shape: ', h_mo.shape)
        h_bg = h_bg.view(h_bg.size(0), h_bg.size(1), -1)
        h_id = h_id.view(h_id.size(0), h_id.size(1), -1)
        h_mo = h_mo.view(h_mo.size(0), h_mo.size(1), -1)
        # logger.debug('view out h_bg shape: ', h_bg.shape)
        # logger.debug('view out h_id shape: ', h_id.shape)
        # logger.debug('view out h_mo shape: ', h_mo.shape)

        h = torch.cat([h_bg, h_id, h_mo], dim=-1)
        logger.debug('cat out h shape: ', h.shape)
        h = h.type(x.dtype)

        return h


if __name__ == '__main__':

    unet_path = '../../config/small_unet.yaml'
    unet_config = OmegaConf.load(unet_path).unet_config
    print(unet_config)
    logger.debug('cuda: ', torch.torch.cuda.device_count())
    num_frames = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug('in this project, use device: ', device)
    video_x = torch.randn(num_frames, 4, 2048).to(device)
    timesteps = torch.tensor([0, 1, 2, 3, 4]).to(device)
    unet = UNetModel(**unet_config).to(device)
    output = unet(video_x, timesteps=timesteps)
    logger.debug('show output shape: ', output.shape)
    

