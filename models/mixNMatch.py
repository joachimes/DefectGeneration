import torch
from pytorch_lightning import LightningModule
from models.mixNMatch_utils.train_first_stage import define_optimizers, Trainer


class MixNMatch(LightningModule):
    def __init__(self, img_size, channels, timesteps=200, batch_size=8, **kwargs) -> None:
        super(MixNMatch, self).__init__()
        self.model = Trainer(self.logger.save_dir)


    def _common_step(self, batch, batch_idx, optimizer_idx):
        loss = 0
        batch_imgs, *_ = batch
        if optimizer_idx == 0:
            loss = self.losses(batch_imgs)
        elif optimizer_idx == 1:
            loss = self.losses(batch_imgs)
        elif optimizer_idx == 2:
            loss = self.losses(batch_imgs)
        return {'loss': loss}

    
    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        res = {'train_avg_loss': avg_loss}
        return res
        

    def val_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        res = {'val_avg_loss': avg_loss}
        self.log_img()
        return res


    def testing_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        res = {'test_avg_loss': avg_loss}
        return res


    def configure_optimizers(self):
        optimizersD, optimizerBD, optimizerGE = define_optimizers(self.model.netG, self.model.netsD, self.model.BD, self.model.encoder)
        return [optimizersD, optimizerBD, optimizerGE]
