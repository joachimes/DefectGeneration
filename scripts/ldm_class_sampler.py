import os
import torch
import hydra 
import os.path as osp
import numpy as np

import sys
sys.path.append('../DefectGeneration')

from pytorch_lightning import seed_everything
from dataset.vial_loader import VialDataModule

from glob import glob
from tqdm import tqdm
from PIL import Image

from omegaconf import DictConfig, OmegaConf
from models import LatentDiffusion
from main import fill_none
from models.ldm_utils.ddim import DDIMSampler

@hydra.main(version_base=None, config_path="../config", config_name="ldm_init")
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
    defects_names = dm.defect_names

    log_folder = 'tb_logs'
    model_name = f"{cfg.state.model_name}_CAM{cfg.dataset.camera}"
    
    get_first_letters = lambda hparam: ''.join([word[:3] for word in hparam.split('_')])
    version_name = f"{'_'.join([get_first_letters(hp) for hp in cfg.state.version_hparams])}"
    
    get_cfg_value = lambda x:  [cfg[cfg_index][x] for cfg_index in cfg if x in cfg[cfg_index]]
    version = f"{'_'.join([str(get_cfg_value(x)[0]) for x in cfg.state.version_hparams])}"
    
    model_path = osp.join(log_folder, model_name, version_name, version)
    
    weight_path = None
    if cfg.state.load and len((model_dir := os.listdir(model_path))) > 0:
        weight_file = [f_name for f_name in model_dir if 'model_' in f_name][-1]
        weight_path = osp.join(model_path, weight_file) 
        print(f'Restoring model from {weight_file} from {version_name}')
    else:
        ''
    accepted_models = [LatentDiffusion.__name__]
    assert cfg.state.model_name in accepted_models, 'Model not supported' 
    
    model = LatentDiffusion(**cfg.model).load_from_checkpoint(weight_path)
    sample(cfg, model, defects_names, version)


def sample(cfg, model, defect_info, model_version, num_images=1000, batch_size=16, steps=50):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    images = glob(osp.join('..', 'img', 'input', '*'))
    masks = glob(osp.join('..', 'img', 'mask', '*'))
    outdir = f'../img/{model_version}/CAM2'
    sampler = DDIMSampler(model)
    
    with torch.no_grad():
        with model.ema_scope():
            for defect_name in defect_info:
                print(f'Sampling defect {defect_name}')
                defect_num = defect_info[defect_name]
                batch = model.get_learned_conditioning(torch.ones(batch_size).to(device) * defect_num)
                outpath = osp.join(outdir, defect_name, 'diffusion', 'train', 'images')
                os.makedirs(outpath, exist_ok=True)
                
                for i in tqdm(range(0, num_images, batch_size)):
                    shape = (cfg.model.channels, cfg.model.image_size, cfg.model.image_size)
                    samples, _ = sampler.sample(S=steps, batch_size=batch_size, shape=shape, conditioning=batch, verbose=False)
                    samples =  samples.cpu().numpy().transpose(0,2,3,1)[0]*255
                    for j in range(i, i+batch_size):
                        Image.fromarray(samples.astype(np.uint8)).save(osp.join(outpath, f'{j}.jpg'))


if __name__ == "__main__":
    main()
