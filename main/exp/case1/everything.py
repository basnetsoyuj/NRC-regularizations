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
    array = [0, 1, 3, 7, 11, 12, 13, 15, 19, 23, 24, 25, 27, 31, 35, 36, 37, 39, 43, 47, 48, 49, 51, 55, 59, 60, 61, 63, 67, 71, 72, 73, 75, 79, 83, 84, 85, 87, 91, 95, 96, 97, 99, 103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 216, 217, 218, 219, 223, 227, 228, 229, 230, 231, 235, 239, 240, 241, 242, 243, 247, 251, 252, 253, 254, 255, 259, 263, 264, 265, 266, 267, 271, 275, 276, 277, 278, 279, 283, 287, 288, 289, 290, 291, 295, 299, 300, 301, 302, 303, 307, 311, 312, 313, 314, 315, 319, 323, 324, 325, 327, 331, 335, 336, 337, 339, 343, 347, 348, 349, 351, 355, 359, 360, 361, 363, 367, 371, 372, 373, 375, 379, 383, 384, 385, 387, 391, 395, 396, 397, 399, 403, 407, 408, 409, 411, 415, 419, 420, 421, 423, 427, 431, 432, 433, 435, 439, 443, 444, 445, 447, 451, 455, 456, 457, 459, 463, 467, 468, 469, 471, 475, 479, 480, 481, 483, 487, 491, 492, 493, 495, 499, 503, 504, 505, 507, 511, 515, 516, 517, 519, 523, 527, 528, 529, 531, 535, 539, 540, 541, 542, 543, 547, 551, 552, 553, 554, 555, 559, 563, 564, 565, 566, 567, 571, 575, 576, 577, 578, 579, 583, 587, 588, 589, 590, 591, 595, 599, 600, 601, 602, 603, 607, 611, 612, 613, 614, 615, 619, 623, 624, 625, 626, 627, 631, 635, 636, 637, 638, 639, 643, 647]

    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=0)
    args = parser.parse_args()
    setting = array[args.setting]

    lamWvalues = np.logspace(-5, -1, 9, endpoint=True)

    settings = [
        'seed', 'S', [1, 2],
        'env', 'E', ['swimmer', 'reacher', 'hopper'],
        'mode', 'M', ['null'],

        'max_epochs', '', [1],
        'batch_size', '', [256],
        'data_size', '', [0],
        'arch', '', ['256-R-256-R-256-R|T'],
        'normalize', '', ['none'],

        'optimizer', '', ['sgd'],
        'lamH', 'T', ['non-UFM'], # , 'UFM'],
        'lamW', 'W', lamWvalues.tolist(),
        'lr', 'lr', [1e-2],

        'eval_freq', '', [100],

        'whitening', 'Wh', ['none', 'zca', 'standardization'],
        # 'whitening', 'Wh', ['none'],
        'single_task', 'ST', [0, 1, 2, None], # None for multitask
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
    
    if config.env == 'swimmer':
        config.max_epochs = int(2e5)
        config.data_size = 1000
    elif config.env == 'reacher':
        config.max_epochs = int(2e5)
        config.lamW = 1.5 * config.lamW
        if config.lamH != -1:
            print("shouldn't happen")
            config.lamH = 1.5 * config.lamW
        config.data_size = 1000
    elif config.env == 'hopper':
        config.max_epochs = 40000
        config.data_size = 10000


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
    config.project = 'everything'
    config.group = 'everything'
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    run_BC(config)


if __name__ == '__main__':
    main()
