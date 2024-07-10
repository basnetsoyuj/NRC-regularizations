import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
# from datasets import load_dataset
import pandas as pd


ENVIRONMENT_NAMES = ['reacher', 'swimmer']
SAVE_FOLDER = './dataset/mujoco'

for env in ENVIRONMENT_NAMES:
    for split in ['train', 'test']:
        file_name = f'{env}_{split}.parquet'
        file_path = os.path.join(SAVE_FOLDER, file_name)

        df = pd.read_parquet(file_path)
        num_eps = df.shape[0]
        disordered_data = df.to_dict()

        obs = []
        actions = []
        rewards = []
        for e in range(num_eps):
            eq_obs = disordered_data['continuous_observations'][e]
            eq_actions = disordered_data['continuous_actions'][e]
            eq_rewards = disordered_data['rewards'][e]
            obs.extend([eq_obs[t][None, ...] for t in range(eq_obs.shape[0])])
            actions.extend([eq_actions[t][None, ...] for t in range(eq_actions.shape[0])])
            rewards.extend([eq_rewards[t] for t in range(eq_rewards.shape[0])])

        dataset = {
            'observations': np.concatenate(obs, axis=0),
            'actions': np.concatenate(actions, axis=0),
            'rewards': np.array(rewards)
        }

        for k, v in dataset.items():
            print(k, v.shape)

        save_file_name = f'{env}_{split}.pkl'
        save_file_path = os.path.join(SAVE_FOLDER, save_file_name)
        with open(save_file_path, 'wb') as file:
            pickle.dump(dataset, file)

for env in ENVIRONMENT_NAMES:
    train_path = os.path.join(SAVE_FOLDER, f'{env}_train.pkl')
    test_path = os.path.join(SAVE_FOLDER, f'{env}_test.pkl')
    try:
        with open(train_path, 'rb') as file1, open(test_path, 'rb') as file2:
            train_data = pickle.load(file1)
            test_data = pickle.load(file2)
        print(f'Load dataset {env}.')
    except Exception as e:
        print(e)

    full_data = {k: np.concatenate((train_data[k], test_data[k]), axis=0) for k in train_data.keys()}
    # full_data['rewards'] = np.concatenate((train_data['rewards'], test_data['rewards']))

    for k, v in full_data.items():
        print(k, v.shape)

    save_file_name = f'{env}.pkl'
    save_file_path = os.path.join(SAVE_FOLDER, save_file_name)
    with open(save_file_path, 'wb') as file:
        pickle.dump(full_data, file)





