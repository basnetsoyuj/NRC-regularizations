import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd


ENVIRONMENT_NAMES = ['reacher', 'swimmer']
SAVE_FOLDER = './dataset/mujoco'

for env in ENVIRONMENT_NAMES:
    file_name = f'{env}.pkl'
    file_path = os.path.join(SAVE_FOLDER, file_name)

    try:
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)
        print(f'Load dataset {file_name}.')
    except Exception as e:
        print(e)

    obs = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    print(f'Data size: {obs.shape[0]}')
    print(f'Observations dimension: {obs.shape[1]}; Actions dimension: {actions.shape[1]}')
    print(f"Average Episode Return: {rewards.reshape(-1, 50 if env == 'reacher' else 1000).sum(axis=1).mean()}")

    action_cov = np.cov(actions, rowvar=False)
    print('The covariance matrix is:')
    print(action_cov)
    print('=========================')

    x = actions[:, 0]
    y = actions[:, 1]

    plt.scatter(x, y, s=0.0001)  # Create a scatter plot
    plt.title('Scatter Plot of Data')
    plt.xlabel('X Axis Label')
    plt.ylabel('Y Axis Label')
    plt.grid(True)  # Optional: Add grid for better readability
    plt.show()



