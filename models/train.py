from pytorch_lightning import LightningModule
from models import Efficientnet, DiffusionNet

class LitTrainer(LightningModule):
    def __init__(self, model_name='Efficientnet', lr=1e-3, weight_decay=None, **kwargs) -> None:
        super(LitTrainer, self).__init__()
        accepted_models = [Efficientnet.__name__, DiffusionNet.__name__] # Extendable
        assert model_name in accepted_models, 'Model not supported' 
        self.model_module = eval(model_name)(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay if weight_decay else 0

    
    def configure_optimizers(self):
        return self.model_module.optimizer(self.model_module.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    
    def training_step(self, batch, batch_idx):
        return self.model_module.train_step(batch, batch_idx)
    
    
    def training_epoch_end(self, outputs):
        results = self.model_module.train_epoch_end(outputs)
        for result in results:
            self.logger.experiment.add_scalar(result, results[result],self.current_epoch)

    
    def validation_step(self, batch, batch_idx):
        return self.model_module.val_step(batch, batch_idx)

    
    def validation_epoch_end(self, outputs):
        results = self.model_module.val_epoch_end(outputs)
        for result in results:
            self.logger.experiment.add_scalar(result, results[result],self.current_epoch)
        self.log('val_loss', results['val_avg_loss'])
        return results


    def test_step(self, batch, batch_idx):
        return self.model_module.test_step(batch, batch_idx)


    def test_epoch_end(self, outputs):
        results = self.model_module.testing_end(outputs)
        for result in results:
            self.logger.experiment.add_scalar(result, results[result],self.current_epoch)

        return results