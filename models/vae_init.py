import torch
from torch import nn
from torchvision.utils import  make_grid
from models.train import LitTrainer
import torch.nn.functional as F
from models.utils.autoencoder_utils import Encoder, Decoder

class VariationalAutoencoder(LitTrainer):
    def __init__(self, img_size, channels, latent_dim, dim_mults, batch_size=8, alpha=1, **kwargs) -> None:
        super(VariationalAutoencoder, self).__init__(**kwargs)
        self.channels = channels
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, dim_mults, (img_size, img_size), channels)
        self.decoder = Decoder(latent_dim, dim_mults, (img_size, img_size), channels)

        self.alpha = alpha
        self.recon_loss_criterion = nn.MSELoss()

        
    @torch.no_grad()
    def log_samples(self):
        # TODO: Sample images from dataloader
        output_samples = next(iter(self.test_dataloader()))
        output_sample = output_samples.reshape(-1, 1, 28, 28) #Reshape tensor to stack the images nicely
        output_sample = self.scale_image(output_sample)
        # save_image(output_sample, f"vae_images/epoch_{self.current_epoch+1}.png")

        shape  = (16, self.channels, self.img_size, self.img_size)
        samples = self.p_sample_loop(shape)

        grid = make_grid((samples[-1] + 1) * 0.5, nrow=4)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

        
    def _common_step(self, batch, batch_idx):
        batch_imgs, _, defect = batch

        hidden, mu, log_var = self.encoder(batch_imgs)
        x_out = self.decoder(hidden)
    
        recon_loss = self.recon_loss_criterion(batch_imgs, x_out)
        kl_loss =  (-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)        
        
        loss = recon_loss * self.alpha + kl_loss
        
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'kl_loss': kl_loss, 'recon_loss': recon_loss}

    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        res = {'train_avg_loss': avg_loss}
        return res
        

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        res = {'val_avg_loss': avg_loss}
        
        avg_loss = torch.stack([x['kl_loss'] for x in outputs]).mean()
        res = {'val_kl_loss': avg_loss}
        
        avg_loss = torch.stack([x['recon_loss'] for x in outputs]).mean()
        res = {'val_recon_loss': avg_loss}
        self.log_samples()
        return res


    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        res = {'test_avg_loss': avg_loss}
        
        avg_loss = torch.stack([x['kl_loss'] for x in outputs]).mean()
        res = {'test_kl_loss': avg_loss}
        
        avg_loss = torch.stack([x['recon_loss'] for x in outputs]).mean()
        res = {'test_recon_loss': avg_loss}
        self.log_samples()
        return res
    