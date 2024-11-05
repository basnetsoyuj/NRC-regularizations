import argparse
import os
import pyrallis
import torch

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
        'mode', 'M', ['null'],

        'max_epochs', '', [200000],
        'batch_size', '', [1000],
        'data_size', 'DS', [1000],
        'arch', '', [('256-R-' * 10)[:-1] + '|T'],
        'normalize', '', ['none'],

        'optimizer', '', ['sgd'],
        'lamH', '', [-1],
        'lamW', 'W', [1e-05, 2.51188643e-05, 6.30957344e-05, 0.000158489319, 0.000398107171, 0.001, 0.00251188643, 0.00630957344, 0.0158489319, 0.0398107171, 0.1],
        'lr', 'lr', [1e-2],

        'eval_freq', '', [500],
        'seed', '', [0],

        'env', 'E', ['swimmer'],
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.num_eval_batch = 100
    
    config.lamH = config.lamW
    if config.lamH != -1:
        config.arch = config.arch.replace('-R|', '|')

    if config.env == 'hopper':
        config.data_size = 10000

    # if config.data_size == 1000:
    #     if config.lamW > 0.005:
    #         config.max_epochs = int(8e5)
    #     else:
    #         config.max_epochs = int(8e5)
    #     config.eval_freq = 100
    # elif config.data_size == 5000:
    #     config.max_epochs = int(2e4)
    # elif config.data_size == 10000:
    #     config.max_epochs = int(2e4)
    # elif config.data_size == 100000:
    #     config.max_epochs = int(2e3)

    # if config.mode == 'no_relu':
    #     config.arch = '256-R-256-R-256|T'
    # elif config.mode == 'gelu':
    #     config.arch = '256-R-256-R-256-G|T'

    config.data_folder = './dataset/mujoco/'
    config.project = 'test-run'
    config.group = 'test-exp'
    config.name = f'E{config.env}_W{config.lamW}_H{config.lamH}_A{config.arch}'

    run_BC(config)


if __name__ == '__main__':
    main()
