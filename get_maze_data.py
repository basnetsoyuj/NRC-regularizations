import os

ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
ld_library_path += ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['LD_LIBRARY_PATH'] = ld_library_path
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'

import d4rl
import gym
import pickle

MAZE_ENVIRONMENT_LIST = ['maze2d-open-v0',
                         'maze2d-umaze-v1',
                         'maze2d-medium-v1',
                         'maze2d-large-v1',
                         ]
DATA_SAVE_PATH = './dataset/maze2d'

for env_name in MAZE_ENVIRONMENT_LIST:
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    file_name = '_'.join(env_name.split('-')[:2]) + '.pkl'
    file_path = os.path.join(DATA_SAVE_PATH, file_name)
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(dataset, file)
        print(f'Save dataset {env_name} to location: {file_path}.')
    except Exception as e:
        print(e)

import pickle

with open('hopper_expert.pkl', 'rb') as f:
    dataset = pickle.load(f)

size = dataset['observations'].shape[0]

with open('hopper_train.pkl', 'wb') as f:
    train_split = {k: v[:int(size*0.9), :] for k, v in dataset.item()}
    pickle.dump(train_split, f)
with open('hopper_test.pkl', 'wb') as f:
    test_split = {k: v[int(size*0.9):, :] for k, v in dataset.item()}
    pickle.dump(test_split, f)