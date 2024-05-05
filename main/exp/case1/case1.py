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
        'env', '', ['swimmer', 'reacher'],
        'mode', 'M', ['add_bias', 'no_relu', 'both', 'null'],

        'max_epochs', '', [100],
        'batch_size', '', [256],
        'data_ratio', 'DR', [0.1, 0.01, 0.001],
        'arch', '', ['256-R-256-R|False'],
        'normalize', '', ['none'],

        'lamH', '', [-1],
        'lamW', 'lamW', [1e-1, 8e-2, 6e-2, 4e-2, 2e-2,
                         1e-2, 8e-3, 6e-3, 4e-3, 2e-3,
                         1e-3, 8e-4, 6e-4, 4e-4, 2e-4,
                         1e-4, 5e-5, 1e-5, 1e-6, 0],
        'lr', '', [3e-4],

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
    elif config.mode == 'no_relu':
        config.arch = '256-R-256|False'
    elif config.mode == 'both':
        config.arch = '256-R-256|True'
    # elif config.mode == 'use_tanh':
    #     config.arch = '256-T-256-T|False'
    # elif config.mode == 'normal_y':
    #     config.normalize = 'normal'
    # elif config.mode == 'standard_y':
    #     config.normalize = 'standard'
    # Test
    config.max_epochs = 1

    config.data_folder = '/NC_regression/dataset/mujoco'
    config.project = 'NC_case1'
    config.group = 'test'
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    run_BC(config)


if __name__ == '__main__':
    main()
