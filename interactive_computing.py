import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd

ENVIRONMENT_NAMES = ['swimmer']
SPLIT = 'train'
SAVE_FOLDER = './dataset/mujoco'

for env in ENVIRONMENT_NAMES:
    file_name = f'{env}_{SPLIT}.pkl'
    file_path = os.path.join(SAVE_FOLDER, file_name)

    try:
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)
        print(f'Load dataset {file_name}.')
    except Exception as e:
        print(e)

    use_size = int(dataset['observations'].shape[0] * 0.1)
    obs = dataset['observations'][:use_size, :]
    actions = dataset['actions'][:use_size, :]

    mu11, mu22 = np.square(actions).mean(axis=0)
    mu12 = (actions[:, 0] * actions[:, 1]).mean()

    sqrt = np.sqrt((mu22-mu11)**2 + 4 * mu12**2)
    gamma1 = (mu22 - mu11 + sqrt) / (2 * mu12)
    gamma2 = (mu22 - mu11 - sqrt) / (2 * mu12)

    print(gamma1, gamma2)
