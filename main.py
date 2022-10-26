import os.path as osp
from torch.cuda import is_available
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset.vial_loader import VialDataModule
from models.train import LitTrainer
import tensorboard
import hydra 
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="classifier")
def main(cfg: DictConfig) -> None:
    seed_everything(42)
    print(OmegaConf.to_yaml(cfg))

    log_folder = 'tb_logs'
    model_name = f"{cfg['model']['model_name']}_CAM{cfg['dataset']['camera']}"
    version = str(cfg['dataset']['batch_size'])+'_'+str(cfg['model']['max_epochs'])
    
    dm = VialDataModule(**cfg['dataset'])
    category_level = 'num_classes' if 'num_classes' in cfg['model'] else 'num_defects'
    cfg['model'][category_level] = dm.__getattribute__(category_level)
    model = LitTrainer(**cfg['model'])
    logger = TensorBoardLogger(log_folder, name=model_name, version=version)

    callbacks = []
    callbacks.append(EarlyStopping(patience=10, monitor='val_loss'))
    callbacks.append(ModelCheckpoint(dirpath=osp.join(log_folder, model_name, version)
                                    , monitor='val_loss'
                                    , filename='model'
                                    , verbose=True
                                    , save_top_k=2
                                    , mode='min'))
    trainer = Trainer(
        gpus=1 if is_available() else 0,
        max_epochs=cfg['model']['max_epochs'],
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule=dm)

    return

if __name__ == "__main__":
    main()
