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
import json

from omegaconf import DictConfig, OmegaConf
from models import LatentDiffusion
from main import fill_none
from models.ldm_utils.ddim import DDIMSampler

@hydra.main(version_base=None, config_path="../config", config_name="ldm_hybrid_crop")
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
        model_epoch = f"epoch={cfg.state.load_epoch}" if 'load_epoch' in cfg.state else 'epoch='
        if 'load_epoch' in cfg.state and cfg.state.load_epoch == 'last':
            model_epoch = cfg.state.load_epoch
        weight_file = [f_name for f_name in model_dir if model_epoch in f_name][-1]
        weight_path = osp.join(model_path, weight_file) 
        print(f'Restoring model from {weight_file} from {version}')
    else:
        print(f'No model found in {model_path}')
        exit()
    accepted_models = [LatentDiffusion.__name__]
    assert cfg.state.model_name in accepted_models, 'Model not supported' 
    model = LatentDiffusion(**cfg.model)
    model.load_state_dict(torch.load(weight_path)['state_dict'], strict=False)
    sample(cfg, model, defects_names, version)

def make_batch(image, bbox, shape, device):
    if image != None:
        image = np.array(Image.open(image).convert("RGB"))
        image = image.astype(np.float32)/255.0
        image = image[None].transpose(0,3,1,2)
        image = torch.from_numpy(image)


    
    if bbox is not None:
        bbox_points = bbox['bbox']
        label = torch.zeros((1, 1, shape[0], shape[1]))
        label[:, :, int(bbox_points[1]):int(bbox_points[1]+bbox_points[3]), int(bbox_points[0]):int(bbox_points[0]+bbox_points[2])] = 1
    else:
        label = torch.zeros((1, 1, image.shape[2], image.shape[3]))
    batch = {"image": image, "mask": label}
    for k in batch:
        if batch[k] is not None:
            batch[k] = batch[k].to(device=device)
            batch[k] = batch[k]*2.0-1.0
    return batch

def iterate_all_folders_and_get_bbox_info(root_dir, camera=2, setting='train'):
    bbox_info = []
    for root, _, files in tqdm(os.walk(root_dir)):
        if f'CAM{camera}' not in root or f'{os.sep}{setting}' not in root:
            continue
        for file in files:
            if file.endswith(".json"):
                with open(f'{root}{os.sep}{file}', 'r') as f:
                    data = json.load(f)
                    image_shape = [data['images'][0]['height'], data['images'][0]['width']]
                    for annotation in data['annotations']:
                        bbox_info.append({'bbox': annotation['bbox'], 'shape': image_shape})
    return bbox_info        


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def sample(cfg, model, defect_info, model_version, num_images=2000, batch_size=4, steps=50):
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    outdir = f'/nn-seidenader-gentofte/TJSD/VisData/DiffusionSlerpSampler/CAM{cfg.dataset.camera}'
    
    img_shape = (448,512)
    latent_img_shape = img_shape
    spatial_multiplier = model.spatial_cond_stage_model.multiplier
    for _ in range(model.spatial_cond_stage_model.n_stages):
        latent_img_shape = (int(latent_img_shape[0]* spatial_multiplier), int(latent_img_shape[1]*spatial_multiplier))
    with torch.no_grad():
        with model.ema_scope():
            for origin in [ 'Real']:
                indir = f'/nn-seidenader-gentofte/TJSD/VisData/{origin}/CAM{cfg.dataset.camera}'
                
                # iterate over all folders in indir
                for defect_name in sorted(os.listdir(indir)):
                    bbox_info = np.asarray(iterate_all_folders_and_get_bbox_info(osp.join(indir, defect_name)))
                    
                    print(f'Sampling defect {defect_name}')
                    try:
                        defect_num = defect_info[defect_name]
                        source_defect = defect_info['synthetic_'+defect_name]
                    except:
                        continue
                    outpath = osp.join(outdir, defect_name, model_version+f'_sampler_slerp_steps={steps}', 'train', 'images')
                    os.makedirs(outpath, exist_ok=True)

                    batch_class = torch.tensor([defect_num for _ in range(batch_size)]).to(device)
                    batch_class = torch.tensor([defect_num for _ in range(batch_size)]).to(device)
                    source_class = torch.tensor([source_defect for _ in range(batch_size)]).to(device)
                    for i in tqdm(range(0, num_images, batch_size)):
                        batch = {'c_crossattn': batch_class, 'c_concat': torch.zeros((batch_size, 1, img_shape[0], img_shape[1])).to(device)}
                        source_batch = {'c_crossattn': source_class, 'c_concat': torch.zeros((batch_size, 1, img_shape[0], img_shape[1])).to(device)}
                        target_batch = {'c_crossattn': batch_class, 'c_concat': torch.zeros((batch_size, 1, img_shape[0], img_shape[1])).to(device)}

                        if len(bbox_info) == 0:
                            bboxes_coordinates = []
                        else:
                            bboxes_coordinates = bbox_info[np.random.randint(0, len(bbox_info), size=batch_size)]
                                                
                        for j, bbox in enumerate(bboxes_coordinates):
                            output = make_batch(None, bbox, img_shape, device)
                            batch['c_concat'][j] = output['mask']
                        bbox_masks = batch['c_concat']
                        
                        target_input = model.get_learned_conditioning(target_batch)
                        source_input = model.get_learned_conditioning(source_batch)
                        batch_input = model.get_learned_conditioning(batch)
                        slerp_t = np.random.uniform(0.4, 1.0)
                        batch_input['c_crossattn'][0] = slerp(slerp_t, source_input['c_crossattn'][0], target_input['c_crossattn'][0])
                        
                        eta = np.random.uniform(0.1, 0.9)
                        latent_shape = (cfg.model.channels, latent_img_shape[0], latent_img_shape[1])
                        # batch_input = model.get_learned_conditioning(batch)
                        samples, _ = sampler.sample(S=steps, batch_size=batch_size, shape=latent_shape, conditioning=batch_input, verbose=False, eta=eta)
                        samples = model.decode_first_stage(samples)
                        samples = torch.clamp((samples+1.0)/2.0,min=0.0, max=1.0)
                        samples = samples.cpu().numpy().transpose(0,2,3,1) * 255
                        if len(bboxes_coordinates) != 0:
                            bbox_masks = torch.clamp((bbox_masks+1.0)/2.0,min=0.0, max=1.0)
                            bbox_masks = bbox_masks.cpu().numpy().transpose(0,2,3,1) * 255
                            label_path = outpath.replace('images', 'labels')
                            os.makedirs(label_path, exist_ok=True)
                        for j in range(i, i+batch_size):
                            if len(bboxes_coordinates) != 0:
                                # save label which one contains 1 channel
                                # Image.from
                                Image.fromarray(np.concatenate([bbox_masks[j%batch_size].astype(np.uint8),bbox_masks[j%batch_size].astype(np.uint8),bbox_masks[j%batch_size].astype(np.uint8)],axis=2)).save(osp.join(label_path, f'{j:04d}_label.jpg'))
                            Image.fromarray(samples[j%batch_size].astype(np.uint8)).save(osp.join(outpath, f'{j:04d}_{eta}_{slerp_t}.jpg'))
                            print(f'Saved image {j:04d}.jpg')


if __name__ == "__main__":
    main()
