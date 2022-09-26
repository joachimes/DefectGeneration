

import torch
from pytorch_lightning import LightningDataModule


class LitTrainer(LightningDataModule):
    def __init__(self, **kwargs) -> None:
        pass

    def configure_optimizers(self):
        pass

    def Loss(self):
        pass

    def training_step(self, batch, batch_idx):
        out = self.model.forward(batch)
        loss = self.Loss(out, batch.y)
        result = {'loss': loss}
        return result
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        res = {'train_avg_loss': avg_loss}
        self.logger.experiment.log_metrics(res)

        

    def validation_step(self, batch, batch_idx):
        out = self.model.forward(batch)
        loss = self.Loss(out, batch.y)
        res = {'val_loss':loss}
        return res

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        res = {'val_avg_loss': avg_loss}
        self.logger.experiment.log_metrics(res)
        self.log('val_loss', avg_loss)
        return res