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
        'mode', 'M', ['null', 'no_relu', 'gelu'],

        'max_epochs', '', [250],
        'batch_size', '', [256],
        'data_size', 'DS', [1, 10],
        'arch', '', ['256-R-256-R|T'],
        'normalize', '', ['none'],

        'optimizer', 'O', ['sgd'],
        'lamH', '', [-1],
        'lamW', 'W', [1e-1, 8e-2, 6e-2, 4e-2, 2e-2,
                      1e-2, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3],
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
    config.max_epochs = 300 if config.env == 'swimmer' else 4000
    if config.optimizer == 'sgd':
        if config.env == 'reacher':
            config.max_epochs = int(5e6) if config.data_size == 10 else int(1e7)
        else:
            config.max_epochs = int(2e5) if config.data_size == 10 else int(2e6)

        config.eval_freq = 5000 if config.env == 'swimmer' else int(5e4)

    if config.mode == 'no_relu':
        config.arch = '256-R-256|T'
    elif config.mode == 'gelu':
        config.arch = '256-G-256-G|T'

    config.data_folder = '/NC_regression/dataset/mujoco'
    config.project = 'NC_ok_sgd'
    config.group = 'new_nrc3'
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    run_BC(config)


if __name__ == '__main__':
    main()
