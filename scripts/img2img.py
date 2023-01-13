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
from einops import repeat

from omegaconf import DictConfig, OmegaConf
from models import LatentDiffusion
from main import fill_none
from models. ldm_utils. ddim import DDIMSampler

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

    log_folder = 'tb_logs'
    model_name = f"{cfg.state.model_name}_CAM{cfg.dataset.camera}"
    
    get_first_letters = lambda hparam: ''.join([word[:3] for word in hparam.split('_')])
    version_name = f"{'_'.join([get_first_letters(hp) for hp in cfg.state.version_hparams])}"
    
    get_cfg_value = lambda x:  [cfg[cfg_index][x] for cfg_index in cfg if x in cfg[cfg_index]]
    version = f"{'_'.join([str(get_cfg_value(x)[0]) for x in cfg.state.version_hparams])}"
    
    version = 'Synthetic_generative_32_256_8_15'
    
    cfg.model.num_defects = 15
    model_path = osp.join(log_folder, model_name, version_name, version)
    
    weight_path = None
    if cfg.state.load and len((model_dir := os.listdir(model_path))) > 0:
        target_model = f'epoch={cfg.state.load_epoch}' if cfg.state.load_epoch else 'last'
        weight_file = [f_name for f_name in model_dir if target_model in f_name][-1]
        weight_path = osp.join(model_path, weight_file) 
        print(f'Restoring model from {weight_file} from {version}')
    else:
        print(f'No model found in {model_path}')
        exit()
    accepted_models = [LatentDiffusion.__name__]
    assert cfg.state.model_name in accepted_models, 'Model not supported' 
    model = LatentDiffusion(**cfg.model)
    model.load_state_dict(torch.load(weight_path)['state_dict'], strict=False)
    sample(cfg, model, dm.defect_names, version)



def make_batch(image, device):
    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)


    batch = {"image": image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


def sample(cfg, model, defect_info, model_version, num_images=5, batch_size=1, steps=50, n_row=2):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    indir = f'/nn-seidenader-gentofte/TJSD/VisData/Real/CAM{cfg.dataset.camera}/Good/'
    images = glob(osp.join(indir, '**', '**', '**', '**', '*'))
    images = np.random.choice(images, num_images, replace=False)
    outdir = f'../img/{cfg.state.model_name}'
    outdir = f'/nn-seidenader-gentofte/TJSD/VisData/diffusion/CAM{cfg.dataset.camera}'
    sampler = DDIMSampler(model)
    sampler.ddpm_num_timesteps = steps
    with torch.no_grad():
        with model.ema_scope():
            for defect_name in defect_info:
                defect_num = defect_info[defect_name]
                batch = model.get_learned_conditioning(torch.tensor([defect_num for _ in range(batch_size)]).to(device))
                outpath = osp.join(outdir, defect_name, model_version, 'train', 'images')
                os.makedirs(outpath, exist_ok=True)
                
                for i, image in tqdm(enumerate(images)):
                    img_batch = make_batch(image, device=device)
                    shape = (cfg.model.channels, cfg.model.image_size, cfg.model.image_size)
                    
                    x0 = model.first_stage_model.encode(img_batch["image"])
                    for step in range(1, steps):
                        # encode masked image and concat downsampled mask
                        n_row = min(x0.shape[0], n_row)
                        t = repeat(torch.tensor([step]), '1 -> b', b=n_row)
                        t = t.to(device).long()
                        noise = torch.randn_like(x0)
                        x_T = model.q_sample(x_start=x0, t=t, noise=noise)
                        
                        samples_ddim, _ = sampler.sample(S=step, batch_size=batch_size
                                                        , shape=shape, conditioning=batch
                                                        , verbose=False, x_T=x_T)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)

                        predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                                    min=0.0, max=1.0)

                        
                        pred_img = predicted_image.cpu().numpy().transpose(0,2,3,1)[0]*255
                        Image.fromarray(pred_img.astype(np.uint8)).save(osp.join(outpath, f'{i}_step-{step}of-{steps}.jpg'))
                    image = torch.clamp((img_batch["image"]+1.0)/2.0,
                                        min=0.0, max=1.0)
                    img = image.cpu().numpy().transpose(0,2,3,1)[0]*255
                    Image.fromarray(img.astype(np.uint8)).save(osp.join(outpath, f'{i}_og.jpg'))



if __name__ == "__main__":
    main()
