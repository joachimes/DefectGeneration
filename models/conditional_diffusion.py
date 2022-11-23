import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.utils.conditional_UNet import UNetConditional
from models.diffusion_init import DiffusionNet


class ConditionalDiffusionNet(DiffusionNet):
    def __init__(self, num_defects=14, **kwargs) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_defects
        self.model = UNetConditional(num_classes=self.num_classes, **kwargs)


    def p_losses(self, x_start, t, labels, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, labels)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss


    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, shape):
        b = shape[0]
        labels = torch.arange(b).long().to(self.device) % self.num_classes
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=self.device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sampler(img, labels, torch.full((b,), i, device=self.device, dtype=torch.long), i)
            imgs.append(img)
        return imgs
    

    def p_sampler(self, x, labels, t, t_index) -> torch.Tensor:
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t, labels) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 


    def _common_step(self, batch, batch_idx):
        batch_imgs, _, labels = batch
        t = torch.randint(0, self.timesteps, (batch_imgs.shape[0],), device=self.device)
        loss = self.p_losses(batch_imgs, t, labels, loss_type="l2")
        return {'loss': loss}


    
    

