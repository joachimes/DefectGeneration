import os
import torch
import hydra 
import os.path as osp
import numpy as np

from pytorch_lightning import seed_everything
from dataset.vial_loader import VialDataModule

from glob import glob
from tqdm import tqdm
from PIL import Image

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
    sample(cfg, model)



def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


def sample(cfg, model):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    images = glob(osp.join('..', 'img', 'input', '*'))
    masks = glob(osp.join('..', 'img', 'mask', '*'))
    outdir = f'../img/{cfg.state.model_name}'
    os.makedirs(outdir, exist_ok=True)
    sampler = DDIMSampler(model)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                outpath = os.path.join(outdir, os.path.split(image)[1])
                batch = make_batch(image, mask, device=device)

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:])
                c = torch.cat((c, cc), dim=1)

                shape = (c.shape[1]-1,)+c.shape[2:]
                samples_ddim, _ = sampler.sample(S=cfg.model.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)



if __name__ == "__main__":
    main()
