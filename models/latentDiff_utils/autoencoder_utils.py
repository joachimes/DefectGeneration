import numpy as np

import torch
from torch import nn
from functools import partial

from models.utils.diffusion_utils import ResnetBlock, Downsample, Upsample, PreNorm, Attention, LinearAttention, Residual, exists, default, SinusoidalPositionEmbeddings, Block

def Normalize(dim, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=dim, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class StableEncoder(nn.Module):
    def __init__(self, *, init_dim, out_ch, dim_mult=(1,2,4,8), resnet_block_groups,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, channels,
                 img_size, z_channels, double_z=True, num_res_blocks=2,
                 **ignore_kwargs):
        super().__init__()
        self.ch = init_dim
        self.temb_ch = 0
        self.num_resolutions = len(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = img_size
        self.in_channels = channels
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # downsampling
        self.conv_in = nn.Conv2d(channels,
                                self.ch,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        curr_res = img_size
        in_ch_mult = (1,)+tuple(dim_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = init_dim*in_ch_mult[i_level]
            block_out = init_dim*dim_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(block_klass(dim=block_in,
                                    dim_out=block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(Residual(PreNorm(block_in, LinearAttention(block_in))))

                # attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = block_klass(dim=block_in,
                                       dim_out=block_in)
        self.mid.attn_1 = Residual(PreNorm(block_in, Attention(block_in)))
        self.mid.block_2 = block_klass(dim=block_in,
                                       dim_out=block_in)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class StableDecoder(nn.Module):
    def __init__(self, *, init_dim, out_ch, dim_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, channels, resnet_block_groups,
                 img_size, z_channels, give_pre_end=False, tanh_out=False, **ignorekwargs):
        super().__init__()
        self.ch = init_dim
        self.temb_ch = 0
        self.num_resolutions = len(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = img_size
        self.in_channels = channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(dim_mult)
        block_in = init_dim*dim_mult[self.num_resolutions-1]
        curr_res = img_size // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = block_klass(dim=block_in,
                                       dim_out=block_in)
        self.mid.attn_1 = Residual(PreNorm(block_in, Attention(block_in)))
        self.mid.block_2 = block_klass(dim=block_in,
                                       dim_out=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = init_dim*dim_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(block_klass(dim=block_in,
                                         dim_out=block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    Residual(PreNorm(block_in, LinearAttention(block_in)))
                    # attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h





class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

