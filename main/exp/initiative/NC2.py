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
    setting = args.setting

    settings = [
        'env', 'E', ['swimmer', 'reacher'],
        'mode', 'M', ['add_bias', 'del_relu', 'use_tanh', 'normal_y', 'standard_y', 'null'],

        'max_epochs', 'T', [200],
        'batch_size', '', [256],
        'data_ratio', 'DR', [1, 0.1],
        'arch', '', ['256-R-256-R|False'],
        'normalize', '', ['none'],

        'reg_coff_H', 'lamH', [-1, 1e-2, 1e-3, 1e-4],
        'reg_coff_W', 'lamW', [1e-2, 1e-3, 1e-4],
        'lr', 'lr', [3e-4],

        'eval_freq', '', [1],
        'seed', '', [0]
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.num_eval_batch = 100

    if config.mode == 'add_bias':
        config.arch = '256-R-256-R|True'
    elif config.mode == 'del_relu':
        config.arch = '256-R-256|False'
    elif config.mode == 'use_tanh':
        config.arch = '256-T-256-T|False'
    elif config.mode == 'normal_y':
        config.normalize = 'normal'
    elif config.mode == 'standard_y':
        config.normalize = 'standard'

    config.data_folder = '/NC_regression/dataset/mujoco'
    config.group = 'explore_NC2'
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    run_BC(config)


if __name__ == '__main__':
    main()
