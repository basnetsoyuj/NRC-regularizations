import os
import torch
import pyrallis
import argparse

from main.BC import TrainConfig, run_BC
from main.utils import get_setting_dt

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=0)
    args = parser.parse_args()

    settings = [
        {'env': 'reacher',
         'data_size': 1000,
         'arch': '256-R-256-R-256-R|T',
         'max_epochs': int(1.2e6),
         'lamW': 1.5e-3},

        {'env': 'swimmer',
         'data_size': 1000,
         'arch': '256-R-256-R-256-R|T',
         'max_epochs': int(1.2e6),
         'lamW': 1e-2},

        {'env': 'hopper',
         'data_size': 10000,
         'arch': '256-R-256-R-256|T',
         'max_epochs': int(1.2e5),
         'lamW': 9.5e-4},
    ]
    
    new_settings = []
    whitenings = ['pca', 'zca', 'none']
    to_whiten = ['all', 'actions', 'states']
    for i in range(3):
        white_setting = settings[i].copy()
        for whitening in whitenings:
            for tw in to_whiten:
                white_setting['whitening'] = whitening
                white_setting['to_whiten'] = tw
                new_settings.append(white_setting.copy())
                if whitening == 'none':
                    break        
    
    settings = new_settings
    actual_setting = settings[args.setting]

    """replace values & global setup"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    print(config.device)
    config.lamH = -1
    config.lr = 1e-2

    config.eval_freq = 100

    config.data_folder = 'dataset/mujoco'
    config.project = 'NC_explore'
    config.group = 'gamma'
    hyper2logname = {
        'env': 'E',
        'data_size': 'DS',
        'lamW': 'wd',
        'max_epochs': 'Eps',
        'whitening': 'W',
        'to_whiten': 'TW',
    }
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])
    print(config.name)
    run_BC(config)


if __name__ == '__main__':
    main()