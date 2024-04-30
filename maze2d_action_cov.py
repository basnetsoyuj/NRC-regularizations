import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

DATASET_NAMES = ['maze2d_open.pkl',
                 'maze2d_umaze.pkl',
                 'maze2d_medium.pkl',
                 'maze2d_large.pkl',
                 ]
SAVE_FOLDER = './dataset/maze2d'
for file_name in DATASET_NAMES:
    file_path = os.path.join(SAVE_FOLDER, file_name)
    try:
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)
        print(f'Load dataset {file_name}.')
    except Exception as e:
        print(e)

    obs = dataset['observations']
    actions = dataset['actions']
    print(f'Data size: {obs.shape[0]}')
    print(f'Observations dimension: {obs.shape[1]}; Actions dimension: {actions.shape[1]}')

    action_cov = np.cov(actions, rowvar=False)
    print('The covariance matrix is:')
    print(action_cov)
    print('=========================')

    x = actions[:, 0]
    y = actions[:, 1]

    plt.scatter(x, y, s=0.0004)  # Create a scatter plot
    plt.title('Scatter Plot of Data')
    plt.xlabel('X Axis Label')
    plt.ylabel('Y Axis Label')
    plt.grid(True)  # Optional: Add grid for better readability
    plt.show()



