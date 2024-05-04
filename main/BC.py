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
from scipy.linalg import sqrtm
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
    num_eval_batch: int = 100  # Do NC evaluation over a subset of the whole dataset
    data_ratio: float = 1.0  # Reduce Swimmer data, too many of them
    normalize: str = 'none'  # Choose from 'none', 'normal', 'standard'

    arch: str = '256-R-256-R|False'  # Actor architecture
    reg_coff_H: float = 1e-5  # If it is -1, then the model is not UFM.
    reg_coff_W: float = 5e-2
    lr: float = 3e-4

    mode: str = 'null'
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
    b_norm = F.normalize(b, p=2, dim=1) if not torch.equal(a, b) else a_norm

    return torch.mm(a_norm, b_norm.T)


def gram_schmidt(W):
    U = torch.empty_like(W)
    U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)

    proj = torch.dot(U[0, :], W[1, :]) * U[0, :]
    ortho_vector = W[1, :] - proj
    U[1, :] = ortho_vector / torch.norm(ortho_vector, p=2)

    return U


def compute_metrics(metrics, device):
    result = {}
    y = metrics['targets']  # (B,2)
    yhat = metrics['outputs']  # (B,2)
    W = metrics['weights']  # (2,256)
    H = metrics['embeddings']  # (B,256)

    result['prediction_error'] = F.mse_loss(y, yhat).item()
    result['cosSIM_y_yhat'] = cosine_similarity_gpu(y, yhat).mean().item()

    # WWT
    WWT = W @ W.T
    result['W11'] = WWT[0, 0].item()
    result['W22'] = WWT[1, 1].item()
    result['W12'] = WWT[0, 1].item()
    result['cosSIM_W12'] = F.cosine_similarity(W[0], W[1], dim=0).item()
    result['WWT_norm'] = torch.norm(WWT).item()
    del WWT

    # H_pca = pca_for_H.fit_transform(H_np)
    # H_reconstruct = pca_for_H.inverse_transform(H_pca)
    # result['PCA_reconstruction_error'] = np.square(H_np - H_reconstruct).sum(axis=1).mean().item()
    # del H_reconstruct

    # Cosine similarity of Y and H post PCA
    # H_pca_tensor = torch.tensor(H_pca, dtype=torch.float32).to(device)
    # del H_pca
    # result['cosSIM_y_Hpca'] = F.cosine_similarity(H_pca_tensor, y, dim=1).mean().item()
    # del H_pca_tensor

    # NC1
    H_np = H.cpu().numpy()
    pca_for_H = PCA(n_components=2)
    pca_for_H.fit(H_np)
    H_pca = pca_for_H.components_[:2, :]  # First two principal components

    try:
        inverse_mat = np.linalg.inv(H_pca @ H_pca.T)
    except Exception as e:
        print(e)
        result['NC1'] = -1
    else:
        H_proj_PCA = (H_pca.T @ inverse_mat @ H_pca @ H_np.T).T
        result['NC1'] = np.linalg.norm(H_np - H_proj_PCA, ord='fro').item()
    del H_np
    del H_proj_PCA
    del H_pca
    del pca_for_H

    # NC3
    try:
        inverse_mat = torch.inverse(W @ W.T)
    except Exception as e:
        print(e)
        result['NC3'] = -1
    else:
        H_proj_W = (W.T @ inverse_mat @ W @ H.T).T
        result['NC3'] = torch.norm(H - H_proj_W).item()

    del H_proj_W

    # MSE between cosine similarities of embeddings and targets with norm
    cos_H_norm = cosine_similarity_gpu(H, H)
    result['cosSIM_H'] = cos_H_norm.fill_diagonal_(float('nan')).nanmean().item()
    cos_y_norm = cosine_similarity_gpu(y, y)
    indices = torch.triu_indices(cos_H_norm.size(0), cos_H_norm.size(0), offset=1)
    cos_H_norm = cos_H_norm[indices[0], indices[1]]
    cos_y_norm = cos_y_norm[indices[0], indices[1]]
    result['MSE_cosSIM_y_H_norm'] = F.mse_loss(cos_H_norm, cos_y_norm).item()
    del cos_H_norm
    del cos_y_norm

    # MSE between cosine similarities of embeddings and targets
    cos_H = torch.mm(H, H.T)
    cos_y = torch.mm(y, y.T)
    indices = torch.triu_indices(cos_H.size(0), cos_H.size(0), offset=1)
    cos_H = cos_H[indices[0], indices[1]]
    cos_y = cos_y[indices[0], indices[1]]
    result['MSE_cosSIM_y_H'] = F.mse_loss(cos_H, cos_y).item()
    del cos_H
    del cos_y
    del indices

    # MSE between cosine similarities of PCA embeddings and targets
    # cos_H_pca = torch.mm(H_pca_norm, H_pca_norm.transpose(0, 1))
    # indices = torch.triu_indices(cos_H_pca.size(0), cos_H_pca.size(0), offset=1)
    # upper_tri_embeddings_pca = cos_H_pca[indices[0], indices[1]]
    # result['mse_cos_sim_PCA'] = F.mse_loss(upper_tri_embeddings_pca, upper_tri_targets).item()

    # Projection error with Gram-Schmidt
    U = gram_schmidt(W)
    P_E = torch.mm(U.T, U)
    H_proj = torch.mm(H, P_E)
    # H_projected_E_norm = F.normalize(torch.tensor(H_projected_E).float().to(device), p=2, dim=1)
    result['proj_error_H2W'] = F.mse_loss(H_proj, H).item()
    del H_proj

    # Cosine similarity of Y and H with H2W
    H_coordinates = torch.mm(F.normalize(H), U.T)
    result['cosSIM_y_H2W'] = F.cosine_similarity(H_coordinates, y).mean().item()

    return result


class MujocoBuffer(Dataset):
    def __init__(
            self,
            data_folder: str,
            env: str,
            split: str,
            data_ratio: float,
            normalize: str,
            device: str = "cpu",
    ):
        self.size = 0
        self.state_dim = 0
        self.action_dim = 0

        self.states, self.actions = None, None
        self._load_dataset(data_folder, env, split, data_ratio, normalize)

        self.device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def _load_dataset(self, data_folder: str, env: str, split: str, data_ratio: float, normalize: str):
        file_name = '%s_%s.pkl' % (env, split)
        file_path = os.path.join(data_folder, file_name)
        try:
            with open(file_path, 'rb') as file:
                dataset = pickle.load(file)
                self.size = int(dataset['observations'].shape[0] * data_ratio)
                if env == 'swimmer':  # Swimmer has too many data, so by default we use 10% of them.
                    self.size = int(self.size * 0.1)
                self.states = dataset['observations'][:self.size, :]
                self.actions = dataset['actions'][:self.size, :]
            print('Successfully load dataset from: ', file_path)
        except Exception as e:
            print(e)

        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]

        if normalize == 'none':
            pass
        elif normalize == 'standard':
            mean = self.actions.mean(0)
            std = np.maximum(self.actions.std(0), 1e-4)
            self.actions = (self.actions - mean) / std
        elif normalize == 'normal':
            min_action = self.actions.min(axis=0)
            max_action = self.actions.max(axis=0)
            self.actions = (self.actions - min_action) / (max_action - min_action)

        print(f"Dataset size: {self.size}; State Dim: {self.state_dim}; Action_Dim: {self.action_dim}.")

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def get_theory_stats(self):
        Y = self.actions.T
        Sigma = Y @ self.actions / Y.shape[1]

        # Sigma_sqrt = sqrtm(Sigma)
        # eig_vals = np.linalg.eigvalsh(Sigma)
        eig_vals, eig_vecs = np.linalg.eigh(Sigma)
        sqrt_eig_vals = np.sqrt(eig_vals)
        Sigma_sqrt = eig_vecs.dot(np.diag(sqrt_eig_vals)).dot(np.linalg.inv(eig_vecs))

        min_eigval = eig_vals[0]
        max_eigval = eig_vals[-1]

        mu11 = Sigma[0, 0]
        mu12 = Sigma[0, 1]
        mu22 = Sigma[1, 1]

        sqrt = np.sqrt((mu22 - mu11) ** 2 + 4 * mu12 ** 2)
        gamma1 = (mu22 - mu11 + sqrt) / (2 * mu12)
        gamma2 = (mu22 - mu11 - sqrt) / (2 * mu12)

        return {
            'mu11': Sigma[0, 0],
            'mu12': Sigma[0, 1],
            'mu22': Sigma[1, 1],
            'min_eigval': min_eigval,
            'max_eigval': max_eigval,
            'gamma1': gamma1,
            'gamma2': gamma2,
            'sigma11': Sigma_sqrt[0, 0],
            'sigma12': Sigma_sqrt[0, 1],
            'sigma21': Sigma_sqrt[1, 0],
            'sigma22': Sigma_sqrt[1, 1],
        }

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
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, arch: str = '256-R-256-R'):
        super(Actor, self).__init__()

        arch, use_bias = arch.split('|')
        arch = arch[0].split('-')
        use_bias = bool(use_bias[0])

        in_dim = state_dim
        module_list = []
        for i, layer in enumerate(arch):
            if layer == 'R':
                module_list.append(nn.ReLU())
            elif layer == 'T':
                module_list.append(nn.Tanh())
            else:
                out_dim = int(layer)
                module_list.append(nn.Linear(in_dim, out_dim))
                in_dim = out_dim

        self.feature_map = nn.Sequential(*module_list)
        self.W = nn.Linear(in_dim, action_dim, bias=use_bias)

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
            reg_coff_W: float,
            num_eval_batch: int,
            device: str = "cpu",
    ):
        self.actor = actor
        self.actor.train()

        self.actor_optimizer = actor_optimizer

        self.total_it = 0
        self.reg_coff_H = reg_coff_H
        self.reg_coff_W = reg_coff_W
        self.num_eval_batch = num_eval_batch
        self.device = device

    def train(self, batch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        states, actions = batch['states'], batch['actions']

        # Compute actor loss
        if self.reg_coff_H == -1:
            preds = self.actor(states)
            mse_loss = 0.5 * F.mse_loss(preds, actions)
            reg_loss = 0
            for param in self.actor.parameters():
                reg_loss += torch.norm(param) ** 2
            reg_loss = 0.5 * self.reg_coff_W * reg_loss
            train_loss = mse_loss + reg_loss
        else:
            H = self.actor.get_feature(states)
            preds = self.actor.project(H)
            mse_loss = 0.5 * F.mse_loss(preds, actions)
            reg_H_loss = 0.5 * self.reg_coff_H * (torch.norm(H, p=2) ** 2) / H.shape[0]
            reg_W_loss = 0.5 * self.reg_coff_W * torch.norm(self.actor.W.weight) ** 2
            train_loss = mse_loss + reg_H_loss + reg_W_loss

        log_dict["train_mse_loss"] = mse_loss.item()
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
        normalize=config.normalize,
        device=config.device
    )
    val_dataset = MujocoBuffer(
        data_folder=config.data_folder,
        env=config.env,
        split='test',
        data_ratio=config.data_ratio,
        normalize=config.normalize,
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

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "reg_coff_H": config.reg_coff_H,
        "reg_coff_W": config.reg_coff_W,
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

    # TODO: fix and optimize wandb log.
    train_theory_stats = train_dataset.get_theory_stats()
    val_theory_states = val_dataset.get_theory_stats()
    if config.reg_coff_H != -1:
        for d in [train_theory_stats, val_theory_states]:
            d['A11'] = (config.reg_coff_H / config.reg_coff_W) ** 0.5 * d['sigma11'] - config.reg_coff_H
            d['A22'] = (config.reg_coff_H / config.reg_coff_W) ** 0.5 * d['sigma22'] - config.reg_coff_H
            d['A12'] = (config.reg_coff_H / config.reg_coff_W) ** 0.5 * d['sigma12']

    train_log = trainer.NC_eval(train_loader)
    val_log = trainer.NC_eval(val_loader)
    wandb.log({'train': train_log,
               'validation': val_log,
               'trainConstant': train_theory_stats,
               'valConstant': val_theory_states
               })

    for epoch in range(config.max_epochs):
        epoch_train_loss = 0
        count = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.max_epochs} Training"):
            log_dict = trainer.train(batch)
            epoch_train_loss += log_dict['train_mse_loss']
            count += 1
            # wandb.log(trainer.train(batch), step=trainer.total_it)
        epoch_train_loss /= count
        if (epoch + 1) % config.eval_freq == 0:
            train_log = trainer.NC_eval(train_loader)
            val_log = trainer.NC_eval(val_loader)
            wandb.log({'train': train_log,
                       'validation': val_log,
                       'train_loss': epoch_train_loss,
                       'trainConstant': train_theory_stats,
                       'valConstant': val_theory_states
                       })

    # NC2
    W = actor.W.weight.detach().clone().cpu().numpy()
    WWT = W @ W.T
    WWT_normalized = WWT / np.linalg.norm(WWT)
    min_eigval = train_theory_stats['min_eigval']
    Sigma_sqrt = np.array([train_theory_stats[k] for k in ['sigma11', 'sigma12', 'sigma21', 'sigma22']]).reshape(2, 2)
    c_to_plot = np.linspace(0, min_eigval, num=5000)
    NC2_to_plot = []
    for c in c_to_plot:
        c_sqrt = c ** 0.5
        A = Sigma_sqrt - c_sqrt * np.eye(2)
        A_normalized = A / np.linalg.norm(A)
        NC2_to_plot.append(np.linalg.norm(WWT_normalized - A_normalized).item())

    data = [[x, y] for (x, y) in zip(c_to_plot, NC2_to_plot)]
    table = wandb.Table(data=data, columns=["c", "NC2"])
    wandb.log(
        {
            "NC2(c)": wandb.plot.line(
                table, "c", "NC2", title="NC2 as a Function of c"
            )
        }
    )

