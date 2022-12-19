import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from torchvision.utils import make_grid
from packaging import version

from models.train import LitTrainer
from models.latentDiff_utils.quantize import VectorQuantizer2 as VectorQuantizer
from models.latentDiff_utils.autoencoder_utils import StableEncoder, StableDecoder
from models.latentDiff_utils.VQVAEloss import VQLPIPSWithDiscriminator



class VQModel(LitTrainer):
    def __init__(self,
                 AEcfg,
                 n_embed,
                 losscfg,
                 latent_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.embed_dim = latent_dim
        self.n_embed = n_embed
        self.encoder = StableEncoder(**AEcfg)
        self.decoder = StableDecoder(**AEcfg)
        if losscfg != 'identity':
            self.loss = VQLPIPSWithDiscriminator(n_classes=n_embed, **losscfg)
        self.quantize = VectorQuantizer(n_embed, latent_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(AEcfg["z_channels"], latent_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(latent_dim, AEcfg["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.fixed_train_imgs = None
        self.fixed_val_imgs = None

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    # def get_input(self, batch, k):
    #     x = batch[k]
    #     if len(x.shape) == 3:
    #         x = x[..., None]
    #     x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
    #     if self.batch_resize_range is not None:
    #         lower_size = self.batch_resize_range[0]
    #         upper_size = self.batch_resize_range[1]
    #         if self.global_step <= 4:
    #             # do the first few batches with max size to avoid later oom
    #             new_resize = upper_size
    #         else:
    #             new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
    #         if new_resize != x.shape[2]:
    #             x = F.interpolate(x, size=new_resize, mode="bicubic")
    #         x = x.detach()
    #     return x

    @torch.no_grad()
    def log_samples(self):

        if self.fixed_train_imgs == None:
            # concatenate multiple validation images from different batches 
            self.fixed_train_imgs = torch.cat([next(iter(self.trainer._data_connector._train_dataloader_source.dataloader()))[0] for i in range(3)], dim=0)[:16]
            self.fixed_val_imgs = torch.cat([next(iter(self.trainer._data_connector._val_dataloader_source.dataloader()))[0] for i in range(3)], dim=0)[:16]
            grid = make_grid((self.fixed_train_imgs[:,:3] + 1) * 0.5, nrow=4)
            self.logger.experiment.add_image(f'vq_imgs/real_train', grid, 0)
            grid = make_grid((self.fixed_val_imgs[:,:3] + 1) * 0.5, nrow=4)
            self.logger.experiment.add_image(f'vq_imgs/real_val', grid, 0)
            if self.fixed_train_imgs.shape[1] == 4:
                grid = make_grid((self.fixed_train_imgs[:, 3] + 1) * 0.5, nrow=4)
                self.logger.experiment.add_image(f'vq_imgs/real_label_train', grid, 0)
                grid = make_grid((self.fixed_val_imgs[:, 3] + 1) * 0.5, nrow=4)
                self.logger.experiment.add_image(f'vq_imgs/real_label_val', grid, 0)
                

        xrec, _ = self(self.fixed_train_imgs.to(self.device))
        xrec_val, _ = self(self.fixed_val_imgs.to(self.device))

        grid = make_grid((xrec[:, :3] + 1) * 0.5, nrow=4)
        self.logger.experiment.add_image(f'vq_imgs/reconstructred_train', grid, self.global_step)

        grid = make_grid((xrec_val[:, :3] + 1) * 0.5, nrow=4)
        self.logger.experiment.add_image(f'vq_imgs/reconstructred_val', grid, self.global_step)

        if self.fixed_train_imgs.shape[1] == 4:
            
            grid = make_grid((xrec[:, 3] + 1) * 0.5, nrow=4)
            self.logger.experiment.add_image(f'vq_imgs/reconstructred_label_train', grid, self.global_step)

            grid = make_grid((xrec_val[:, 3] + 1) * 0.5, nrow=4)
            self.logger.experiment.add_image(f'vq_imgs/reconstructred_label_val', grid, self.global_step)



    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        batch_imgs, *_ = batch

        xrec, qloss, ind = self(batch_imgs, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, batch_imgs, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            if (self.global_step < 1000 and self.global_step % 100 == 0) or self.global_step % 2000 == 0:
                print(f'logging samples at step {self.global_step}')
                self.log_samples()
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, batch_imgs, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss


    def validation_step(self, batch, batch_idx):
        batch_imgs, _, classes  = batch

        xrec, qloss, ind = self(batch_imgs, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, batch_imgs, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val",
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, batch_imgs, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val",
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val/rec_loss"]
        self.log(f"val_rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def validation_epoch_end(self, outputs):
        self.log_samples()

    def configure_optimizers(self):
        lr_d = self.lr
        lr_g = self.lr_g_factor*self.lr
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        batch_imgs, *_ = batch

        batch_imgs = batch_imgs.to(self.device)
        if only_inputs:
            log["inputs"] = batch_imgs
            return log
        xrec, _ = self(batch_imgs)
        # if batch_imgs.shape[1] > 3:
        #     # colorize with random projection
        #     assert xrec.shape[1] > 3
        #     batch_imgs = self.to_rgb(batch_imgs)
        #     xrec = self.to_rgb(xrec)
        log["inputs"] = batch_imgs
        log["reconstructions"] = xrec

        return log

    # def to_rgb(self, x):
    #     assert self.image_key == "segmentation"
    #     if not hasattr(self, "colorize"):
    #         self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
    #     x = F.conv2d(x, weight=self.colorize)
    #     x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
    #     return x


class VQModelInterface(VQModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.Identity() # dummy loss

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec