import torch
from torch import nn
from torchvision.utils import  make_grid
from models.train import LitTrainer
import torch.nn.functional as F
from models.latentDiff_utils.autoencoder_utils import StableEncoder, StableDecoder, DiagonalGaussianDistribution
from models.latentDiff_utils.VAEloss import LPIPSWithDiscriminator

class VariationalAutoEncoder(LitTrainer):
    def __init__(self, latent_dim, AEcfg, losscfg, batch_size=8, **kwargs) -> None:
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.batch_size = batch_size
        
        self.latent_dim = latent_dim
        self.encoder = StableEncoder(**AEcfg)
        self.decoder = StableDecoder(**AEcfg)

        self.loss = LPIPSWithDiscriminator(**losscfg)

        assert AEcfg["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*AEcfg["z_channels"], 2*latent_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(latent_dim, AEcfg["z_channels"], 1)
        self.embed_dim = latent_dim

        self.fixed_imgs = None


    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior


    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)
    

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior


    @torch.no_grad()
    def log_samples(self):

        if self.fixed_imgs == None:
            self.fixed_imgs, *_ = next(iter(self.trainer._data_connector._train_dataloader_source.dataloader()))
            grid = make_grid((self.fixed_imgs[:16] + 1) * 0.5, nrow=4)
            self.logger.experiment.add_image(f'reconstructred_images', grid, 0)

        xrec, posterior = self(self.fixed_imgs[:16].to(self.device))

        grid = make_grid((xrec + 1) * 0.5, nrow=4)
        self.logger.experiment.add_image(f'reconstructred_images', grid, self.current_epoch)

        rnd_samples = self.decode(torch.randn_like(posterior.sample()))
        grid = make_grid((rnd_samples + 1) * 0.5, nrow=4)
        self.logger.experiment.add_image(f'samples_images', grid, self.current_epoch)

        
    def training_step(self, batch, batch_idx, optimizer_idx):
        batch_imgs, *_ = batch

        recon, posterior = self(batch_imgs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(batch_imgs, recon, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(batch_imgs, recon, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss


    def validation_step(self, batch, batch_idx):
        batch_imgs, *_ = batch

        recon, posterior = self(batch_imgs)

        aeloss, log_dict_ae = self.loss(batch_imgs, recon, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(batch_imgs, recon, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val_rec_loss", log_dict_ae["val/rec_loss"], sync_dist=True)
        self.log_dict(log_dict_ae, sync_dist=True)
        self.log_dict(log_dict_disc, sync_dist=True)
        return self.log_dict


    def test_step(self, *args):
        return self.validation_step(*args)
        

    def validation_epoch_end(self, outputs):
        self.log_samples()


    def configure_optimizers(self):
        lr = self.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    

    def get_last_layer(self):
        return self.decoder.conv_out.weight