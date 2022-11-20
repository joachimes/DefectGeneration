import torch
from torchvision.utils import  make_grid
from torch.optim import Adam
from pytorch_lightning import LightningModule
from models.utils.diffusion_utils import cosine_beta_schedule, linear_beta_schedule
from models.utils.UNet import UNet
import torch.nn.functional as F
from tqdm.auto import tqdm

class DiffusionNet(LightningModule):
    def __init__(self, img_size, channels, timesteps=200, batch_size=8, **kwargs) -> None:
        super(DiffusionNet, self).__init__()
        self.channels = channels
        self.img_size = img_size
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.noise_schedule(self.timesteps)
        self.model = UNet(img_size=img_size, channels=channels, **kwargs)

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

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, shape):
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=self.device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sampler( img, torch.full((b,), i, device=self.device, dtype=torch.long), i)
            imgs.append(img)
        return imgs
    
    def p_sampler(self, x, t, t_index) -> torch.Tensor:
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
    @torch.no_grad()
    def log_samples(self):
        
        shape  = (16, self.channels, self.img_size, self.img_size)
        samples = self.p_sample_loop(shape)

        grid = make_grid((samples[-1] + 1) * 0.5, nrow=4)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

        # grid = make_grid(sample['progressive_samples'].reshape(-1, 3, self.img_size, self.img_size), nrow=20)
        # self.logger.experiment.add_image(f'progressive_generated_images', grid, self.current_epoch)
    

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
        self.log_samples()
        return res

    def testing_step(self, batch, batch_idx):
        res = self.train_step(batch, batch_idx)
        return {'test_loss': res['loss']}


    def testing_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        res = {'test_avg_loss': avg_loss}
        self.log_samples()
        return res


    def optimizer(self, parameters, lr, weight_decay):
        return Adam(parameters, lr=lr, weight_decay=weight_decay)


    def num_to_groups(self, num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr




