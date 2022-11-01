import os.path as osp
from torch.cuda import is_available
from pytorch_lightning import Trainer, seed_everything
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
    print(cfg.dataset)
    
    dm = VialDataModule(**cfg.dataset)
    category_level = 'num_classes' if 'num_classes' in cfg.model else 'num_defects'
    cfg.model[category_level] = dm.__getattribute__(category_level)

    log_folder = 'tb_logs'
    model_name = f"{cfg.model.model_name}_CAM{cfg.dataset.camera}"
    version_hparams = ['dataset_type',category_level, 'batch_size', 'max_epochs']
    get_first_letters = lambda hparam: ''.join([word[:3] for word in hparam.split('_')])
    version_name = f"{'_'.join([get_first_letters(hp) for hp in version_hparams])}"
    get_cfg_value = lambda x:  [cfg[cfg_index][x] for cfg_index in cfg if x in cfg[cfg_index]]
    version = f"{'_'.join([str(get_cfg_value(x)[0]) for x in version_hparams])}"
    model = LitTrainer(**cfg.model)
    logger = TensorBoardLogger(log_folder, name=model_name, version=osp.join(version_name, version))

    callbacks = []
    callbacks.append(EarlyStopping(patience=cfg.model.patience, monitor='val_loss'))
    callbacks.append(ModelCheckpoint(dirpath=osp.join(log_folder, model_name, version_name, version)
                                    , monitor='val_loss'
                                    , filename='model'
                                    , verbose=True
                                    , save_top_k=2
                                    , mode='min'))
    trainer = Trainer(
        accelerator="gpu",
        devices=1 if is_available() else 0,
        max_epochs=cfg.model.max_epochs,
        logger=logger,
        callbacks=callbacks,
        # profiler='simple'
    )
    trainer.fit(model, datamodule=dm)

    return

if __name__ == "__main__":
    main()
