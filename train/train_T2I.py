# add the main folder to the path so the modules can be imported without errors
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from re import I
from ddpm import Unet3D, GaussianDiffusion, Trainer
from T2I_Adapter.adapters import Adapter_Medical_Diffusion, T2I_Trainer
from dataset import MRNetDataset, BRATSDataset
import argparse
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from train.get_dataset import get_dataset
import torch
import os


# NCCL_P2P_DISABLE=1 accelerate launch train/train_ddpm.py

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpu_id)


    model = Unet3D(
        dim=cfg.model.diffusion_img_size,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.diffusion_num_channels,
    ).cuda()

    train_dataset, *_ = get_dataset(cfg)

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.diffusion_timesteps,
        loss_type=cfg.model.loss_type,
    ).cuda()


    data = torch.load(cfg.model.diffusion_pt)

    diffusion.load_state_dict(data['ema'])
    diffusion.eval()

    T2I_adapter = Adapter_Medical_Diffusion(zero_conv=cfg.model.zero_conv)

    trainer = T2I_Trainer(
        T2I_model=T2I_adapter,
        T2I_derivate_name=cfg.dataset.mandatory_derivatates[0],
        diffusion_model=diffusion,
        dataset=train_dataset,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        batch_size=cfg.model.batch_size,
        train_lr=cfg.model.train_lr,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_num_steps=cfg.model.train_num_steps,
    )


    if cfg.model.T2I_pt:
        print(f'loading model {cfg.model.T2I_pt}')
        trainer.load(cfg.model.T2I_pt)

    trainer.train()


if __name__ == '__main__':
    run()

