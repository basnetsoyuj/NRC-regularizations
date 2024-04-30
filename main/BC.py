import os
import pickle
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "swimmer"  # OpenAI gym environment name. Choose from 'swimmer' and 'reacher'.
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = 5  # How often (epochs) we evaluate
    # n_episodes: int = 10  # How many episodes run during RL evaluation
    max_epochs: int = 200  # How many epochs to run
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    batch_size: int = 256  # Batch size for all networks
    num_eval_batch: int = 200  # Do NC evaluation over a subset of the whole dataset
    data_ratio: float = 1.0  # Reduce Swimmer data, too many of them

    arch: str = '256-256'  # Actor architecture
    reg_coff_H: float = 1e-5  # If it is -1, then the model is not UFM.
    reg_coff_W: float = 5e-2
    lr: float = 3e-4

    data_folder: str = '/NC_regression/dataset/mujoco'

    # Wandb logging
    project: str = "NC_regression"
    group: str = "test"
    name: str = "test"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def cosine_similarity_gpu(a, b):
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)

    return torch.mm(a_norm, b_norm.transpose(0, 1))


def gram_schmidt(W):
    U = torch.empty_like(W)
    U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)

    proj = torch.dot(U[0, :], W[1, :]) * U[0, :]
    ortho_vector = W[1, :] - proj
    U[1, :] = ortho_vector / torch.norm(ortho_vector, p=2)

    return U


def compute_metrics(metrics, device):
    result = {}
    y = metrics['targets'].to(device)  # (B,2)
    Wh = metrics['outputs'].to(device)  # (B,2)
    W = metrics['weights'].to(device)  # (2,256)
    H = metrics['embeddings'].to(device)  # (B,256)
    print("Y", y.shape)
    print("Wh", Wh.shape)
    print("W", W.shape)
    print("W0", W[0].shape)
    print("H", H.shape)

    result['prediction_error'] = F.mse_loss(Wh, y).item()

    H_norm = F.normalize(H, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    W_norm = F.normalize(W, p=2, dim=1)

    result['W_norm_sq'] = torch.norm(W, p=2).item()
    result['W1_norm_sq'] = torch.dot(W[0], W[0]).item()
    result['W2_norm_sq'] = torch.dot(W[1], W[1]).item()

    # Cosine similarity calculations
    result['cos_sim_y_Wh'] = cosine_similarity_gpu(y, Wh).mean().item()
    # result['cos_sim_W11'] = torch.dot(W[0], W[0]).item()
    result['cos_sim_W12'] = F.cosine_similarity(W[0], W[1], dim=0).item()
    # result['cos_sim_W22'] = torch.dot(W[1], W[1]).item()
    # result['cos_sim_W'] = F.cosine_similarity(W, W).fill_diagonal_(float('nan')).nanmean().item()
    result['cos_sim_H'] = cosine_similarity_gpu(H, H).fill_diagonal_(float('nan')).nanmean().item()

    # H with PCA
    H_norm_np = H.cpu().numpy()
    pca_for_H = PCA(n_components=2)
    H_pca = pca_for_H.fit_transform(H_norm_np)
    H_reconstruct = pca_for_H.inverse_transform(H_pca)
    # result['projection_error_PCA'] = np.mean(np.square(H_norm_np - H_reconstruct))
    result['projection_error_PCA'] = np.square(H_norm_np - H_reconstruct).sum(axis=1).mean().item()

    # Cosine similarity of Y and H post PCA
    H_pca_tensor = torch.tensor(H_pca, dtype=torch.float32).to(device)
    result['cos_sim_y_h_postPCA'] = F.cosine_similarity(H_pca_tensor, y, dim=1).mean().item()
    # H_pca_norm = F.normalize(torch.tensor(H_pca, dtype=torch.float32).to(device), p=2, dim=1)
    # cos_sim_y_h_after_pca = torch.mm(H_pca_norm, y_norm.transpose(0, 1))
    # result['cos_sim_y_h_postPCA'] = cos_sim_y_h_after_pca.diag().mean().item()

    # MSE between cosine similarities of embeddings and targets with norm
    cos_H_norm = torch.mm(H_norm, H_norm.transpose(0, 1))
    cos_y_norm = torch.mm(y_norm, y_norm.transpose(0, 1))
    indices = torch.triu_indices(cos_H_norm.size(0), cos_H_norm.size(0), offset=1)
    upper_tri_embeddings_norm = cos_H_norm[indices[0], indices[1]]
    upper_tri_targets_norm = cos_y_norm[indices[0], indices[1]]
    result['mse_cos_sim_norm'] = F.mse_loss(upper_tri_embeddings_norm, upper_tri_targets_norm).item()

    # MSE between cosine similarities of embeddings and targets
    cos_H = torch.mm(H, H.transpose(0, 1))
    cos_y = torch.mm(y, y.transpose(0, 1))
    indices = torch.triu_indices(cos_H.size(0), cos_H.size(0), offset=1)
    upper_tri_embeddings = cos_H[indices[0], indices[1]]
    upper_tri_targets = cos_y[indices[0], indices[1]]
    result['mse_cos_sim'] = F.mse_loss(upper_tri_embeddings, upper_tri_targets).item()

    # MSE between cosine similarities of PCA embeddings and targets
    # cos_H_pca = torch.mm(H_pca_norm, H_pca_norm.transpose(0, 1))
    # indices = torch.triu_indices(cos_H_pca.size(0), cos_H_pca.size(0), offset=1)
    # upper_tri_embeddings_pca = cos_H_pca[indices[0], indices[1]]
    # result['mse_cos_sim_PCA'] = F.mse_loss(upper_tri_embeddings_pca, upper_tri_targets).item()

    # Projection error with Gram-Schmidt
    U = gram_schmidt(W)
    P_E = torch.mm(U.T, U)
    H_projected_E = torch.mm(H, P_E)
    # H_projected_E_norm = F.normalize(torch.tensor(H_projected_E).float().to(device), p=2, dim=1)
    result['projection_error_H2W_E'] = F.mse_loss(H_projected_E, H).item()

    # Cosine similarity of Y and H with H2W
    H_coordinates = torch.mm(H_norm, U.T)
    H_coordinates_norm = F.normalize(torch.tensor(H_coordinates).float().to(device), p=2, dim=1)
    cos_sim_H2W = torch.mm(H_coordinates_norm, y_norm.transpose(0, 1))
    result['cos_sim_y_h_H2W_E'] = cos_sim_H2W.diag().mean().item()
    return result


class MujocoBuffer(Dataset):
    def __init__(
            self,
            data_folder: str,
            env: str,
            split: str,
            data_ratio,
            device: str = "cpu",
    ):
        self.size = 0
        self.state_dim = 0
        self.action_dim = 0

        self.states, self.actions = None, None
        self._load_dataset(data_folder, env, split, data_ratio)

        self.device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def _load_dataset(self, data_folder: str, env: str, split: str, data_ratio: float):
        file_name = '%s_%s.pkl' % (env, split)
        file_path = os.path.join(data_folder, file_name)
        try:
            with open(file_path, 'rb') as file:
                dataset = pickle.load(file)
                self.size = dataset['observations'].shape[0] * data_ratio
                self.states = dataset['observations'][:self.size, :]
                self.actions = dataset['actions'][:self.size, :]
            print('Successfully load dataset from: ', file_path)
        except Exception as e:
            print(e)

        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        print(f"Dataset size: {self.size}; State Dim: {self.state_dim}; Action_Dim: {self.action_dim}.")

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        return {
            'states': self._to_tensor(states),
            'actions': self._to_tensor(actions)
        }


def set_seed(
        seed: int, env=None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def RL_eval(
        env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, arch: str = '256-256'):
        super(Actor, self).__init__()

        in_dim = state_dim
        module_list = []
        for i, hidden_size in enumerate(arch.split('-')):
            hidden_size = int(hidden_size)
            module_list.append(nn.Linear(in_dim, hidden_size))
            module_list.append(nn.ReLU())
            in_dim = hidden_size
        self.feature_map = nn.Sequential(*module_list)
        self.W = nn.Linear(in_dim, action_dim, bias=False)

        self.max_action = max_action

    def get_feature(self, state: torch.Tensor):
        return self.feature_map(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        H = self.get_feature(state)
        return self.W(H)

    def project(self, feature):
        return self.W(feature)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        # Modified: Clip the actions, since we do not have a tanh in the actor.
        action = self(state).clamp(min=-self.max_action, max=self.max_action)
        return action.cpu().data.numpy().flatten()


class BC:
    def __init__(
            self,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            reg_coff_H: float,
            num_eval_batch: int,
            device: str = "cpu",
    ):
        self.actor = actor
        self.actor.train()

        self.actor_optimizer = actor_optimizer

        self.total_it = 0
        self.reg_coff_H = reg_coff_H
        self.num_eval_batch = num_eval_batch
        self.device = device

    def train(self, batch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        states, actions = batch['states'], batch['actions']

        # Compute actor loss
        if self.reg_coff_H == -1:
            preds = self.actor(states)
            mse_loss = F.mse_loss(preds, actions)
            train_loss = mse_loss
        else:
            H = self.actor.get_feature(states)
            preds = self.actor.project(H)
            mse_loss = F.mse_loss(preds, actions)
            reg_loss = 0.5 * self.reg_coff_H * (torch.norm(H, p=2) ** 2) / H.shape[0]
            train_loss = mse_loss + reg_loss

        log_dict["train_loss"] = mse_loss.item()  # Only log the mse loss as train loss.
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        train_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    @torch.no_grad()
    def NC_eval(self, dataloader):
        self.actor.eval()
        y = torch.empty((0,), device=self.device)
        H = torch.empty((0,), device=self.device)
        Wh = torch.empty((0,), device=self.device)
        W = self.actor.W.weight.detach().clone()

        for i, batch in enumerate(dataloader):
            if i+1 > self.num_eval_batch:
                break
            states, actions = batch['states'], batch['actions']
            features = self.actor.get_feature(states)
            preds = self.actor.project(features)

            y = torch.cat((y, actions), dim=0)
            H = torch.cat((H, features), dim=0)
            Wh = torch.cat((Wh, preds), dim=0)

        res = {'targets': y,
               'embeddings': H,
               'outputs': Wh,
               'weights': W
               }
        log_dict = compute_metrics(res, self.device)
        self.actor.train()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


def run_BC(config: TrainConfig):
    train_dataset = MujocoBuffer(
        data_folder=config.data_folder,
        env=config.env,
        split='train',
        data_ratio=config.data_ratio,
        device=config.device
    )
    val_dataset = MujocoBuffer(
        data_folder=config.data_folder,
        env=config.env,
        split='test',
        data_ratio=config.data_ratio,
        device=config.device
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    # Set seeds
    seed = config.seed
    set_seed(seed)

    state_dim = train_dataset.get_state_dim()
    action_dim = train_dataset.get_action_dim()
    actor = Actor(state_dim, action_dim, arch=config.arch).to(config.device)

    if config.reg_coff_H == -1:
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr, weight_decay=config.reg_coff_W)
    else:
        actor_optimizer = torch.optim.Adam(
            [{'params': actor.feature_map.parameters()},
             {'params': actor.W.parameters(), 'weight_decay': config.reg_coff_W}],
            lr=config.lr
        )

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "reg_coff_H": config.reg_coff_H,
        'num_eval_batch': config.num_eval_batch,
        "device": config.device
    }

    print("---------------------------------------")
    print(f"Training BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize policy
    trainer = BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    train_log = trainer.NC_eval(train_loader)
    val_log = trainer.NC_eval(val_loader)
    wandb.log({'train': train_log, 'validation': val_log}, step=0)

    for epoch in range(config.max_epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.max_epochs} Training"):
            wandb.log(trainer.train(batch), step=trainer.total_it)

        if (epoch + 1) % config.eval_freq == 0:
            train_log = trainer.NC_eval(train_loader)
            val_log = trainer.NC_eval(val_loader)
            wandb.log({'train': train_log, 'validation': val_log}, step=epoch)
