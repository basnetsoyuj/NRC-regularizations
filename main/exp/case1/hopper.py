import os
import torch
import pyrallis
import argparse
import numpy as np

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

    lamWvalues = np.logspace(-5, -1, 9, endpoint=True)

    settings = [
        'seed', 'S', [0],
        'env', 'E', ['hopper'],
        'mode', 'M', ['null'],

        'max_epochs', '', [int(2e5)],
        'batch_size', '', [256],
        'data_size', '', [10000],
        # 'arch', '', ['256-R-256-R-256-R|T'],
        'arch', '', ['256-R-B-256-R-B-256-R-B|T'],
        'normalize', '', ['none'],

        'optimizer', '', ['sgd'],
        'lamH', 'T', ['non-UFM', 'UFM'],
        'lamW', 'W', lamWvalues.tolist(),
        'lr', 'lr', [1e-2],

        'eval_freq', '', [100],

        'whitening', 'Wh', ['none', 'zca', 'standardization'],
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)

    if config.lamH == 'non-UFM':
        config.lamH = -1
    elif config.lamH == 'UFM':
        config.lamH = config.lamW
    
    config.device = DEVICE
    config.num_eval_batch = 100

    # if config.data_size == 1000:
    #     config.max_epochs = int(1.2e6)
    #     config.eval_freq = 100
    # elif config.data_size == 5000:
    #     config.max_epochs = int(2e5)
    # elif config.data_size == 10000:
    #     config.max_epochs = int(8e4)
    # elif config.data_size == 100000:
    #     config.max_epochs = int(8e3)

    # if config.mode == 'no_relu':
    #     config.arch = '256-R-256-R-256|T'
    # elif config.mode == 'gelu':
    #     config.arch = '256-R-256-R-256-G|T'

    config.data_folder = './dataset/mujoco/'
    config.project = 'std_whitening'
    config.group = 'final'
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    run_BC(config)


if __name__ == '__main__':
    main()
