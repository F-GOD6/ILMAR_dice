from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import h5py
import os
from urllib import request
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from learner import Learner
LOG_STD_MIN = -5
LOG_STD_MAX = 2
SCALE_DIAG_MIN_MAX = (LOG_STD_MIN, LOG_STD_MAX)
MEAN_MIN_MAX = (-7, 7)
EPS = np.finfo(np.float32).eps
KEYS = ['observations', 'actions', 'rewards', 'terminals']

import torch
import torch.nn as nn
import torch.nn.functional as F


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))

class TanhActor(nn.Module):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 hidden_size=256, 
                 name='TanhNormalPolicy',
                 mean_range=(-7., 7.), 
                 logstd_range=(-5., 2.), 
                 eps=1e-6, 
                 initial_std_scaler=1.0,
                 activation_fn=nn.ReLU,
                 kernel_initializer=nn.init.kaiming_normal_):
        """
        初始化 TanhActor 网络。

        Args:
            state_dim (int): 状态的维度。
            action_dim (int): 动作的维度。
            hidden_size (int, optional): 隐藏层的神经元数量。默认值为 256。
            name (str, optional): 网络名称。默认值为 'TanhNormalPolicy'。
            mean_range (tuple, optional): 均值的范围。默认值为 (-7., 7.)。
            logstd_range (tuple, optional): 对数标准差的范围。默认值为 (-5., 2.)。
            eps (float, optional): 用于数值稳定性的极小值。默认值为 1e-6。
            initial_std_scaler (float, optional): 初始标准差的缩放因子。默认值为 1.0。
            activation_fn (nn.Module, optional): 激活函数。默认值为 nn.ReLU。
            kernel_initializer (function, optional): 权重初始化函数。默认值为 nn.init.kaiming_normal_。
        """
        super(TanhActor, self).__init__()

        self.action_dim = action_dim
        self.initial_std_scaler = initial_std_scaler
        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

        # 定义 MLP 层
        self.fc_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn()
        )

        # 定义均值和对数标准差的输出层
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        self.fc_logstd = nn.Linear(hidden_size, action_dim)

        # 初始化权重
        self._initialize_weights(kernel_initializer)

    def _initialize_weights(self, initializer):
        """
        初始化网络中的线性层权重。

        Args:
            initializer (function): 权重初始化函数。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                initializer(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs, training=True):
        """
        前向传播，生成动作及其对数概率。

        Args:
            inputs (torch.Tensor): 输入的状态张量。
            training (bool, optional): 是否处于训练模式。默认值为 True。

        Returns:
            tuple: 包含 (tanh(mean), action, log_prob) 和 network_state（None）的元组。
        """
        h = self.fc_layers(inputs)
        
        # 计算均值并限制范围
        mean = self.fc_mean(h)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        
        # 计算对数标准差并限制范围
        logstd = self.fc_logstd(h)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        
        # 计算标准差
        std = torch.exp(logstd) * self.initial_std_scaler

        # 创建独立正态分布
        normal_dist = Normal(mean, std)
        pretanh_action_dist = Independent(normal_dist, 1)

        # 采样预激活动作（使用 rsample 以支持重参数化）
        pretanh_action = pretanh_action_dist.rsample()
        
        # 应用 Tanh 激活函数
        action = torch.tanh(pretanh_action)

        # 计算对数概率
        log_prob = self.log_prob(pretanh_action_dist, pretanh_action)

        # 计算确定性动作（均值经过 Tanh）
        deterministic_action = torch.tanh(mean)

        return (deterministic_action, action, log_prob), None  # network_state 为 None

    def log_prob(self, pretanh_action_dist, pretanh_action):
        """
        计算动作的对数概率。

        Args:
            pretanh_action_dist (Independent): 预激活动作的分布。
            pretanh_action (torch.Tensor): 预激活动作。

        Returns:
            torch.Tensor: 动作的对数概率。
        """
        # 通过 Tanh 变换得到最终动作
        action = torch.tanh(pretanh_action)

        # 计算预激活动作的对数概率
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)

        # 计算 Tanh 变换的雅可比行列式的对数值
        log_det_jacobian = torch.sum(torch.log(1 - action ** 2 + self.eps), dim=-1)

        # 总的对数概率
        log_prob = pretanh_log_prob - log_det_jacobian

        return log_prob

    def get_log_prob(self, states, actions):
        """
        根据状态和动作计算对数概率。

        Args:
            states (torch.Tensor): 一批状态。
            actions (torch.Tensor): 一批动作。

        Returns:
            torch.Tensor: 动作的对数概率。
        """
        h = self.fc_layers(states)

        # 计算均值并限制范围
        mean = self.fc_mean(h)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)

        # 计算对数标准差并限制范围
        logstd = self.fc_logstd(h)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)

        # 计算标准差
        std = torch.exp(logstd) * self.initial_std_scaler

        # 创建独立正态分布
        normal_dist = Normal(mean, std)
        pretanh_action_dist = Independent(normal_dist, 1)

        # 限制动作值以避免数值不稳定
        clipped_actions = torch.clamp(actions, -1 + self.eps, 1 - self.eps)

        # 计算预激活动作（atanh）
        # 检查是否支持 torch.atanh
        if hasattr(torch, 'atanh'):
            pretanh_actions = torch.atanh(clipped_actions)
        else:
            # 如果不支持，手动实现 atanh
            pretanh_actions = atanh(clipped_actions)

        # 计算预激活动作的对数概率
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_actions)

        # 计算 Tanh 变换的雅可比行列式的对数值
        log_det_jacobian = torch.sum(torch.log(1 - actions ** 2 + self.eps), dim=-1)

        # 总的对数概率
        log_probs = pretanh_log_prob - log_det_jacobian

        # 为了避免广播问题，增加一个维度
        log_probs = log_probs.unsqueeze(-1)

        return log_probs

class MetaTanhActor(nn.Module):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 hidden_size=256, 
                 name='TanhNormalPolicy',
                 mean_range=(-7., 7.), 
                 logstd_range=(-5., 2.), 
                 eps=1e-6, 
                 initial_std_scaler=1.0,
                 activation_fn:str = "relu",
                 kernel_initializer=nn.init.kaiming_normal_):
        """
        初始化 TanhActor 网络。

        Args:
            state_dim (int): 状态的维度。
            action_dim (int): 动作的维度。
            hidden_size (int, optional): 隐藏层的神经元数量。默认值为 256。
            name (str, optional): 网络名称。默认值为 'TanhNormalPolicy'。
            mean_range (tuple, optional): 均值的范围。默认值为 (-7., 7.)。
            logstd_range (tuple, optional): 对数标准差的范围。默认值为 (-5., 2.)。
            eps (float, optional): 用于数值稳定性的极小值。默认值为 1e-6。
            initial_std_scaler (float, optional): 初始标准差的缩放因子。默认值为 1.0。
            activation_fn (nn.Module, optional): 激活函数。默认值为 nn.ReLU。
            kernel_initializer (function, optional): 权重初始化函数。默认值为 nn.init.kaiming_normal_。
        """
        super(MetaTanhActor, self).__init__()
        self.action_dim = action_dim
        model_config= [
        ('linear', [hidden_size, state_dim]),
        ('bn', [hidden_size]),
        (activation_fn, [True]),
        ('linear', [hidden_size, hidden_size]),
        ('bn', [hidden_size]),
        (activation_fn, [True]),
        ('linear', [2*action_dim, hidden_size]),
    ]
        self.initial_std_scaler = initial_std_scaler
        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps
        self.net = Learner(model_config)

    def forward(self, inputs, training=True):
        """
        前向传播，生成动作及其对数概率。

        Args:
            inputs (torch.Tensor): 输入的状态张量。
            training (bool, optional): 是否处于训练模式。默认值为 True。

        Returns:
            tuple: 包含 (tanh(mean), action, log_prob) 和 network_state（None）的元组。
        """
        if training:
            self.net.train()  # 否则，如果是训练模式，设置为 train 模式
            means, logstd = self.net(inputs).chunk(2, dim=-1)
        else:
            self.net.eval()  # 如果是测试模式，设置为 eval 模式
            means, logstd = self.net(inputs,bn_training=False).chunk(2, dim=-1)
        means = torch.clamp(means, self.mean_min, self.mean_max)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        std = torch.exp(logstd) * self.initial_std_scaler
        normal_dist = Normal(means, std)
        pretanh_action_dist = Independent(normal_dist, 1)
        pretanh_action = pretanh_action_dist.rsample()
        action = torch.tanh(pretanh_action)
        log_prob = self.log_prob(pretanh_action_dist, pretanh_action)
        deterministic_action = torch.tanh(means)
        self.net.train()
        return (deterministic_action, action, log_prob), None  # network_state 为 None

    def log_prob(self, pretanh_action_dist, pretanh_action):
        """
        计算动作的对数概率。

        Args:
            pretanh_action_dist (Independent): 预激活动作的分布。
            pretanh_action (torch.Tensor): 预激活动作。

        Returns:
            torch.Tensor: 动作的对数概率。
        """
        # 通过 Tanh 变换得到最终动作
        action = torch.tanh(pretanh_action)

        # 计算预激活动作的对数概率
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)

        # 计算 Tanh 变换的雅可比行列式的对数值
        log_det_jacobian = torch.sum(torch.log(1 - action ** 2 + self.eps), dim=-1)

        # 总的对数概率
        log_prob = pretanh_log_prob - log_det_jacobian

        return log_prob

    def get_log_prob(self, states, actions,vars=None):
        """
        根据状态和动作计算对数概率。

        Args:
            states (torch.Tensor): 一批状态。
            actions (torch.Tensor): 一批动作。

        Returns:
            torch.Tensor: 动作的对数概率。
        """
        if vars == None:
            mean, logstd = self.net(states).chunk(2, dim=-1)
        else:
            mean, logstd = self.net(states,vars).chunk(2, dim=-1)
        if not torch.isfinite(mean).all():
            print("States:", states)
            raise ValueError("Means contain NaN or Inf!")

        # 检查 logstd 是否有限，如果不是，则打印 states 并抛出异常
        if not torch.isfinite(logstd).all():
            print("States:", states)
            raise ValueError("Logstd contain NaN or Inf!")
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        # 计算标准差
        std = torch.exp(logstd) * self.initial_std_scaler
        assert torch.isfinite(std).all(), "std contain NaN or Inf!"
        # 创建独立正态分布
        normal_dist = Normal(mean, std)
        pretanh_action_dist = Independent(normal_dist, 1)

        # 限制动作值以避免数值不稳定
        clipped_actions = torch.clamp(actions, -1 + self.eps, 1 - self.eps)

        # 计算预激活动作（atanh）
        # 检查是否支持 torch.atanh
        if hasattr(torch, 'atanh'):
            pretanh_actions = torch.atanh(clipped_actions)
        else:
            # 如果不支持，手动实现 atanh
            pretanh_actions = atanh(clipped_actions)

        # 计算预激活动作的对数概率
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_actions)

        # 计算 Tanh 变换的雅可比行列式的对数值
        log_det_jacobian = torch.sum(torch.log(1 - actions ** 2 + self.eps), dim=-1)

        # 总的对数概率
        log_probs = pretanh_log_prob - log_det_jacobian

        # 为了避免广播问题，增加一个维度
        log_probs = log_probs.unsqueeze(-1)

        return log_probs

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, output_activation_fn=None,
                 use_last_layer_bias=False, output_dim=None, kernel_initializer='he_normal'):
        """
        PyTorch implementation of the Critic network from SAC.

        Args:
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
            hidden_size (int, optional): Number of units in hidden layers. Default is 256.
            output_activation_fn (function, optional): Activation function for the output layer. Default is None.
            use_last_layer_bias (bool, optional): Whether to use bias in the output layer. Default is False.
            output_dim (int, optional): Dimensionality of the output. Default is None (single Q-value).
            kernel_initializer (str, optional): Initializer for weights. Default is 'he_normal'.
        """
        super(Critic, self).__init__()
        
        self.output_dim = output_dim or 1

        # 定义隐藏层
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # 输出层
        if use_last_layer_bias:
            # 使用随机均匀初始化偏置
            last_layer_init = nn.init.uniform_(torch.empty(self.output_dim), -3e-3, 3e-3)
            self.output_layer = nn.Linear(hidden_size, self.output_dim)
            self.output_layer.bias.data = last_layer_init
        else:
            # 不使用偏置
            self.output_layer = nn.Linear(hidden_size, self.output_dim, bias=False)

        # 设置输出激活函数
        self.output_activation_fn = output_activation_fn

        # 权重初始化
        self._initialize_weights(kernel_initializer)

    def _initialize_weights(self, initializer):
        """
        Initialize the weights of the network.

        Args:
            initializer (str): Initializer for weights ('he_normal', etc.).
        """
        if initializer == 'he_normal':
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='linear')
        elif initializer == 'xavier_uniform':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.output_layer.weight)
        else:
            raise ValueError(f"Unsupported kernel_initializer: {initializer}")

        # 初始化偏置为 0
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs):
        """
        Forward pass through the Critic network.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Q-value(s).
        """
        x = inputs  # 拼接状态和动作
        x = F.relu(self.fc1(x))                 # 第一隐藏层
        x = F.relu(self.fc2(x))                 # 第二隐藏层
        x = self.output_layer(x)                # 输出层

        # 如果定义了输出激活函数，则应用它
        if self.output_activation_fn:
            x = self.output_activation_fn(x)

        if self.output_dim == 1:
            x = x.view(-1)  # 如果输出是标量 Q 值，则展平输出

        return x



def load_d4rl_data(dirname, env_id, dataname, num_trajectories, start_idx=0, dtype=np.float32):
    MAX_EPISODE_STEPS = 1000

    original_env_id = env_id
    if env_id in ['Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Ant-v2']:
        env_id = env_id.split('-v2')[0].lower()

    filename = f'{env_id}_{dataname}'
    filepath = os.path.join(dirname, filename + '.hdf5')
    # if not exists
    if not os.path.exists(filepath):
        os.makedirs(dirname, exist_ok=True)
        # Download the dataset
        remote_url = f'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/{filename}.hdf5'
        print(f'Download dataset from {remote_url} into {filepath} ...')
        request.urlretrieve(remote_url, filepath)
        print(f'Done!')

    def get_keys(h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    dataset_file = h5py.File(filepath, 'r')
    dataset_keys = KEYS
    use_timeouts = False
    use_next_obs = False
    if 'timeouts' in get_keys(dataset_file):
        if 'timeouts' not in dataset_keys:
            dataset_keys.append('timeouts')
        use_timeouts = True
    dataset = {k: dataset_file[k][:] for k in dataset_keys}
    dataset_file.close()
    N = dataset['observations'].shape[0]
    init_obs_, init_action_, obs_, action_, next_obs_, rew_, done_ = [], [], [], [], [], [], []
    episode_steps = 0
    num_episodes = 0
    for i in range(N - 1):
        if env_id == 'ant':
            obs = dataset['observations'][i][:27]
            if use_next_obs:
                next_obs = dataset['next_observations'][i][:27]
            else:
                next_obs = dataset['observations'][i + 1][:27]
        else:
            obs = dataset['observations'][i]
            if use_next_obs:
                next_obs = dataset['next_observations'][i]
            else:
                next_obs = dataset['observations'][i + 1]
        action = dataset['actions'][i]
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            is_final_timestep = dataset['timeouts'][i]
        else:
            is_final_timestep = (episode_steps == MAX_EPISODE_STEPS - 1)

        if is_final_timestep:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break
            continue

        if num_episodes >= start_idx:
            if episode_steps == 0:
                init_obs_.append(obs)
            obs_.append(obs)
            next_obs_.append(next_obs)
            action_.append(action)
            done_.append(done_bool)

        episode_steps += 1
        if done_bool:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break

    env = gym.make(original_env_id)
    if env.action_space.dtype == int:
        action_ = np.eye(env.action_space.n)[np.array(action_, dtype=np.int)]  # integer to one-hot encoding

    print(f'{num_episodes} trajectories are sampled')
    return np.array(init_obs_, dtype=dtype), np.array(obs_, dtype=dtype), np.array(action_, dtype=dtype), np.array(
        next_obs_, dtype=dtype), np.array(done_)


def add_absorbing_states(expert_states, expert_actions, expert_next_states,
                         expert_dones, env, dtype=np.float32):
    """Adds absorbing states to trajectories.
    Args:
      expert_states: A numpy array with expert states.
      expert_actions: A numpy array with expert states.
      expert_next_states: A numpy array with expert states.
      expert_dones: A numpy array with expert states.
      env: A gym environment.
    Returns:
        Numpy arrays that contain states, actions, next_states and dones.
    """

    # First add 0 indicator to all non-absorbing states.
    expert_states = np.pad(expert_states, ((0, 0), (0, 1)), mode='constant')
    expert_next_states = np.pad(
        expert_next_states, ((0, 0), (0, 1)), mode='constant')

    expert_states = [x for x in expert_states]
    expert_next_states = [x for x in expert_next_states]
    expert_actions = [x for x in expert_actions]
    expert_dones = [x for x in expert_dones]

    # Add absorbing states.
    i = 0
    current_len = 0
    while i < len(expert_states):
        current_len += 1
        if expert_dones[i] and current_len < env._max_episode_steps:  # pylint: disable=protected-access
            current_len = 0
            expert_states.insert(i + 1, env.get_absorbing_state())
            expert_next_states[i] = env.get_absorbing_state()
            expert_next_states.insert(i + 1, env.get_absorbing_state())
            action_dim = env.action_space.n if env.action_space.dtype == int else env.action_space.shape[0]
            expert_actions.insert(i + 1, np.zeros((action_dim,), dtype=dtype))
            expert_dones[i] = 0.0
            expert_dones.insert(i + 1, 1.0)
            i += 1
        i += 1

    expert_states = np.stack(expert_states)
    expert_next_states = np.stack(expert_next_states)
    expert_actions = np.stack(expert_actions)
    expert_dones = np.stack(expert_dones)

    return expert_states.astype(dtype), expert_actions.astype(dtype), expert_next_states.astype(dtype), expert_dones.astype(dtype)

    
class RolloutBuffer:
    """
    Rollout buffer that often used in training RL agents

    Parameters
    ----------
    buffer_size: int
        size of the buffer
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    mix: int
        the buffer will be mixed using these time of data
    """
    def __init__(
            self,
            buffer_size: int,
            state_shape: int,
            action_shape: int,
            mix: int = 1
    ):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.buffer = {
            'states': np.empty((self.total_size, state_shape), dtype=np.float32),
            'actions': np.empty((self.total_size, action_shape), dtype=np.float32),
            'rewards': np.empty((self.total_size, 1), dtype=np.float32),
            'dones': np.empty((self.total_size, 1), dtype=np.float32),
            'next_states': np.empty((self.total_size, state_shape), dtype=np.float32)
        }

    def append(
            self,
            state: np.array,
            action: np.array,
            reward: float,
            done: bool,
            next_state: np.array
    ):
        """
        Save a transition in the buffer

        Parameters
        ----------
        state: np.array
            current state
        action: np.array
            action taken in the state
        reward: float
            reward of the s-a pair
        done: bool
            whether the state is the end of the episode
        log_pi: float
            log(\pi(a|s))
        next_state: np.array
            next states that the s-a pair transferred to
        """
        self.buffer['states'][self._p] = state
        self.buffer['actions'][self._p] = action
        self.buffer['rewards'][self._p] = reward
        self.buffer['dones'][self._p] = done
        self.buffer['next_states'][self._p] = next_state

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def sample(
            self,
            batch_size: int,
            indices: np.array = None,
            to_numpy: bool = False
    ):
        """
        Sample data from the buffer

        Parameters
        ----------
        batch_size: int
            batch size
        indices: np.array, optional
            specific indices to sample from the buffer
        to_numpy: bool, optional
            whether to return the sampled data in numpy format

        Returns
        -------
        states: np.array
        actions: np.array
        rewards: np.array
        dones: np.array
        log_pis: np.array
        next_states: np.array
        """
        if indices is None:
            indices = np.random.randint(low=0, high=self._n, size=batch_size)

        return self.buffer['states'][indices],self.buffer['actions'][indices],self.buffer['next_states'][indices],self.buffer['dones'][indices],
