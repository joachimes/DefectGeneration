from functools import partial
import os.path as osp
import torch
from torch import nn
from torchvision.utils import  make_grid
from torch.optim import Adam
from pytorch_lightning import LightningModule
import numpy as np
from models.utils import *

class Unet(LightningModule):
    def __init__(
        self,
        img_size,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        **kwargs,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, img_size // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: img_size * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = img_size * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(img_size),
                nn.Linear(img_size, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(img_size, img_size), nn.Conv2d(img_size, out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


class DiffusionNet(LightningModule):
    def __init__(self, img_size, channels, timesteps=200, batch_size=8, **kwargs) -> None:
        super(DiffusionNet, self).__init__()
        self.channels = channels
        self.img_size = img_size
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.noise_schedule(self.timesteps)
        self.model = Unet(img_size=img_size, channels=channels, **kwargs)

    def noise_schedule(self, timesteps=200):

        # define beta schedule
        self.betas = cosine_beta_schedule(timesteps=timesteps)

        # define alphas 
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


    def p_losses(self, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss


    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)
    

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # # Algorithm 2 (including returning all images)
    # @torch.no_grad()
    # def p_sample_loop(self, model, shape):
    #     b = shape[0]
    #     # start from pure noise (for each example in the batch)
    #     img = torch.randn(shape, device=self.device)
    #     imgs = []

    #     for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
    #         img = self.p_sample(model, img, torch.full((b,), i, device=self.device, dtype=torch.long), i)
    #         imgs.append(img.permute(0,2,3,1).cpu().numpy())
    #     return torch.FloatTensor(imgs)

    # @torch.no_grad()
    # def sample(self, model, image_size, batch_size=16, channels=3):
    #     return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
    # @torch.no_grad()
    def p_mean_variance(self, x, t, clip_denoised, return_pred_x0):
        model_output = self.model(x, t)

        # # Learned or fixed variance?
        # if self.model_var_type == 'learned':
        #     model_output, log_var = torch.split(model_output, 2, dim=-1)
        #     var                   = torch.exp(log_var)

        # elif self.model_var_type in ['fixedsmall', 'fixedlarge']:

        #     # below: only log_variance is used in the KL computations
        #     var, log_var = {
        #         # for 'fixedlarge', we set the initial (log-)variance like so to get a better decoder log likelihood
        #         'fixedlarge': (self.betas, torch.log(torch.cat((self.posterior_variance[1].view(1, 1),
        #                                                         self.betas[1:].view(-1, 1)), 0)).view(-1)),
        #         'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
        #     }[self.model_var_type]

        #     var     = self.extract(var, t, x.shape) * torch.ones_like(x)
        #     log_var = self.extract(log_var, t, x.shape) * torch.ones_like(x)
        # else:
        #     raise NotImplementedError(self.model_var_type)

        var = None
        log_var = None
        # Mean parameterization
        _maybe_clip = lambda x_: (x_.clamp(min=-1, max=1) if clip_denoised else x_)
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
 
        pred_x0 = _maybe_clip(model_output)
         # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        mean = sqrt_recip_alphas_t * (
            x - betas_t * pred_x0 / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] != 0:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            mean += torch.sqrt(posterior_variance_t) * noise 
        
        if return_pred_x0:
            return mean, var, log_var, pred_x0
        else:
            return mean, var, log_var, None

    # def p_sample(self, model, x, t, t_index):
    #     betas_t = self.extract(self.betas, t, x.shape)
    #     sqrt_one_minus_alphas_cumprod_t = self.extract(
    #         self.sqrt_one_minus_alphas_cumprod, t, x.shape
    #     )
    #     sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
    #     # Equation 11 in the paper
    #     # Use our model (noise predictor) to predict the mean
    #     model_mean = sqrt_recip_alphas_t * (
    #         x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    #     )
    #     if t_index == 0:
    #         return model_mean
    #     else:
    #         posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
    #         noise = torch.randn_like(x)
    #         # Algorithm 2 line 4:
    #         return model_mean + torch.sqrt(posterior_variance_t) * noise 

    

    @torch.no_grad()
    def p_sample(self, x, t, noise_fn, clip_denoised=True, return_pred_x0=False):

        mean, _, log_var, pred_x0 = self.p_mean_variance( x, t, clip_denoised, return_pred_x0=True)
        noise                     = noise_fn(x.shape, dtype=x.dtype).to(x.device)

        # shape        = [x.shape[0]] + [1] * (x.ndim - 1)
        # nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape).to(x.device)
        # sample       = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

        return (mean, pred_x0) if return_pred_x0 else mean
        
    
    @torch.no_grad()
    def p_sample_loop_progressive(self, shape, noise_fn=torch.randn, include_x0_pred_freq=50):

        img = noise_fn(shape, dtype=torch.float32, device=self.device)

        num_recorded_x0_pred = self.timesteps // include_x0_pred_freq
        x0_preds_            = torch.zeros((shape[0], num_recorded_x0_pred, *shape[1:]), dtype=torch.float32, device=self.device)

        for i in reversed(range(self.timesteps)):

            # Sample p(x_{t-1} | x_t) as usual
            img, pred_x0 = self.p_sample(x=img,
                                         t=torch.full((shape[0],), i, dtype=torch.int64, device=self.device),
                                         noise_fn=noise_fn,
                                         return_pred_x0=True)

            # Keep track of prediction of x0
            insert_mask = np.floor(i // include_x0_pred_freq) == torch.arange(num_recorded_x0_pred,
                                                                              dtype=torch.int32,
                                                                              device=self.device)

            insert_mask = insert_mask.to(torch.float32).view(1, num_recorded_x0_pred, *([1] * len(shape[1:])))
            x0_preds_   = insert_mask * pred_x0[:, None, ...] + (1. - insert_mask) * x0_preds_

        return img, x0_preds_

    @torch.no_grad()
    def progressive_samples_fn(self, shape, include_x0_pred_freq=50):
        samples, progressive_samples = self.p_sample_loop_progressive(
            shape=shape,
            noise_fn=torch.randn,
            include_x0_pred_freq=include_x0_pred_freq
        )
        return {'samples': (samples + 1)/2, 'progressive_samples': (progressive_samples + 1)/2}


    def train_step(self, batch, batch_idx):
        batch_imgs, *_ = batch
        t = torch.randint(0, self.timesteps, (batch_imgs.shape[0],), device=self.device)
        loss = self.p_losses(batch_imgs, t, loss_type="huber")

        return {'loss': loss}
    
    
    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        res = {'train_avg_loss': avg_loss}
        return res
        

    def val_step(self, batch, batch_idx):
        res = self.train_step(batch, batch_idx)
        return {'val_loss': res['loss']}


    def val_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        res = {'val_avg_loss': avg_loss}
        self.log_img()
        return res

    def testing_step(self, batch, batch_idx):
        res = self.train_step(batch, batch_idx)
        return {'test_loss': res['loss']}


    def testing_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        res = {'test_avg_loss': avg_loss}
        return res


    def optimizer(self, parameters, lr, weight_decay):
        return Adam(parameters, lr=lr, weight_decay=weight_decay)


    def log_img(self):
        
        shape  = (16, 3, self.img_size, self.img_size)
        sample = self.progressive_samples_fn(shape)

        grid = make_grid(sample['samples'], nrow=4)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

        grid = make_grid(sample['progressive_samples'].reshape(-1, 3, self.img_size, self.img_size), nrow=20)
        self.logger.experiment.add_image(f'progressive_generated_images', grid, self.current_epoch)
    

    def num_to_groups(self, num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr




