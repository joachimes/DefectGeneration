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
from models import Efficientnet, DiffusionNet, ConditionalDiffusionNet, MixNMatch, VAEModel, VQModel, LatentDiffusion


@hydra.main(version_base=None, config_path="config", config_name="classifier")
def main(cfg: DictConfig) -> None:
    seed_everything(42)
    print(OmegaConf.to_yaml(cfg))
    print(cfg.dataset)
    
    dm = VialDataModule(**cfg.dataset)
    for category_attribute in ['num_classes', 'num_defects', 'class_names', 'defect_names']:
        if category_attribute in cfg.model:
            cfg.model[category_attribute] = dm.__getattribute__(category_attribute)
            if 'num_' in category_attribute:
                cfg.state.version_hparams.append(category_attribute)
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
    monitor = cfg.state.monitor if 'monitor' in cfg.state else 'val_loss'
    callbacks.append(EarlyStopping(patience=cfg.model.patience, monitor=monitor))
    callbacks.append(ModelCheckpoint(dirpath=model_path
                                    , monitor= monitor
                                    , filename='model_{epoch}_{val_loss:.3f}'
                                    , verbose=True
                                    , save_top_k=cfg.model.save_top_k if 'save_top_k' in cfg.model else 1
                                    , mode='min'
                                    , save_last=cfg.state.save_last if 'save_last' in cfg.state else False))
    trainer = Trainer(
        accelerator='gpu' if cfg.state.gpu is not None else 'cpu',
        devices=cfg.state.gpu if cfg.state.gpu and is_available() else None,
        # max_epochs=cfg.model.max_epochs,
        max_time={'days': 3},
        logger=logger,
        accumulate_grad_batches=cfg.state.gradient_accum if 'gradient_accum' in cfg.state else None,
        callbacks=callbacks,
        precision=16 if cfg.state.precision == 'mixed' else 32,
        # profiler='simple'
    )
    weight_path = None
    if cfg.state.load and len((model_dir := os.listdir(model_path))) > 0:
        model_epoch = f"epoch={cfg.state.load_epoch}" if cfg.state.load_epoch else 'epoch='
        weight_file = [f_name for f_name in model_dir if model_epoch in f_name][-1]
        weight_path = osp.join(model_path, weight_file) 
     
    accepted_models = [Efficientnet.__name__, DiffusionNet.__name__, ConditionalDiffusionNet.__name__, MixNMatch.__name__, VAEModel.__name__
                        , VQModel.__name__, LatentDiffusion.__name__]
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
