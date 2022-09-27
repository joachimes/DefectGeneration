import os.path as osp
from torch.cuda import is_available
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset.vial_loader import VialDataModule
from models.train import LitTrainer

import hydra 
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    seed_everything(42)
    print(OmegaConf.to_yaml(cfg))

    dm = VialDataModule(**cfg['dataset'])
    batch = next(iter(dm.train_dataloader()))
    out_dim = len(batch)
    return


    logger = CSVLogger(f'logs'
                        , name=f"log_{cfg['model']['model_name']}"
                        , version=str(cfg['batch_size'])+'_'+str(cfg['max_epochs']))
    model = LitTrainer()

    callbacks = []
    callbacks.append(EarlyStopping(patience=10, monitor='val_loss'))
    callbacks.append(ModelCheckpoint(dirpath=osp.join('logs', f"log_{cfg['model']['model_name']}")
                                    , monitor='val_loss'
                                    , filename='model'
                                    , verbose=True
                                    , save_top_k=1
                                    , mode='min'))
    trainer = Trainer(
        gpus=1 if is_available() else 0,
        max_epochs=10,
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
