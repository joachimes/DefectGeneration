import os
import os.path as osp
from torch.cuda import is_available
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset.vial_loader import VialDataModule
import tensorboard
import hydra 
from omegaconf import DictConfig, OmegaConf
from models import Efficientnet, DiffusionNet, ConditionalDiffusionNet, MixNMatch


@hydra.main(version_base=None, config_path="config", config_name="classifier")
def main(cfg: DictConfig) -> None:
    seed_everything(42)
    print(OmegaConf.to_yaml(cfg))
    print(cfg.dataset)
    
    dm = VialDataModule(**cfg.dataset)
    category_num = 'num_classes' if 'num_classes' in cfg.model else 'num_defects'
    cfg.state.version_hparams.append(category_num)
    cfg.model[category_num] = dm.__getattribute__(category_num)
    category_names = 'class_names' if 'class_names' in cfg.model else 'defect_names'
    cfg.model[category_names] = dm.__getattribute__(category_names)
    fill_none(cfg)

    log_folder = 'tb_logs'
    model_name = f"{cfg.state.model_name}_CAM{cfg.dataset.camera}"
    
    get_first_letters = lambda hparam: ''.join([word[:3] for word in hparam.split('_')])
    version_name = f"{'_'.join([get_first_letters(hp) for hp in cfg.state.version_hparams])}"
    
    get_cfg_value = lambda x:  [cfg[cfg_index][x] for cfg_index in cfg if x in cfg[cfg_index]]
    version = f"{'_'.join([str(get_cfg_value(x)[0]) for x in cfg.state.version_hparams])}"
    
    model_path = osp.join(log_folder, model_name, version_name, version)
    
    logger = TensorBoardLogger(log_folder, name=model_name, version=osp.join(version_name, version))
    callbacks = []
    callbacks.append(EarlyStopping(patience=cfg.model.patience, monitor='val_loss'))
    callbacks.append(ModelCheckpoint(dirpath=model_path
                                    , monitor='val_loss'
                                    , filename='model_{epoch}_{val_loss:.3f}'
                                    , verbose=True
                                    , save_top_k=cfg.model.save_top_k if 'save_top_k' in cfg.model else 1
                                    , mode='min'))
    trainer = Trainer(
        accelerator="gpu",
        devices=1 if is_available() else 0,
        max_epochs=cfg.model.max_epochs,
        logger=logger,
        callbacks=callbacks,
        # profiler='simple'
    )
    weight_path = None
    if cfg.state.load and len((model_dir := os.listdir(model_path))) > 0:
        weight_file = [f_name for f_name in model_dir if 'model_' in f_name][-1]
        weight_path = osp.join(model_path, weight_file) 
     
    accepted_models = [Efficientnet.__name__, DiffusionNet.__name__, ConditionalDiffusionNet.__name__, MixNMatch.__name__]
    assert cfg.state.model_name in accepted_models, 'Model not supported' 
    
    model = eval(cfg.state.model_name)(**cfg.model)
    if cfg.state.mode == 'train':
        trainer.fit(model, datamodule=dm, ckpt_path=weight_path)
    else:
        trainer.test(model, datamodule=dm, ckpt_path=weight_path)
    return


def fill_none(cfg:DictConfig):
    for cfg_key, cfg_inner in cfg['dataset'].items():
        if cfg_inner is None and cfg_key in cfg['model']:
            cfg['dataset'][cfg_key] = cfg['model'][cfg_key] 
    
    for cfg_key, cfg_inner in cfg['model'].items():
        if cfg_inner is None and cfg_key in cfg['dataset']:
            cfg['model'][cfg_key] =  cfg['dataset'][cfg_key] 

if __name__ == "__main__":
    main()
