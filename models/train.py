from pytorch_lightning import LightningModule
from torch.optim import Adam

class LitTrainer(LightningModule):
    def __init__(self, lr=1e-3, weight_decay=None, **kwargs) -> None:
        super(LitTrainer, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay if weight_decay else 0
        self.model = None
        self.save_hyperparameters()
    
    def _common_step(self, *args):
        pass
   
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)


    def validation_step(self, *args):
        return self._common_step(*args)

    
    def test_step(self, *args):
        return self._common_step(*args)

    
    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
