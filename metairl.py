import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import utils

EPS = np.finfo(np.float32).eps
EPS2 = 1e-3
def get_device():
    # 检查 CUDA_VISIBLE_DEVICES 环境变量
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    if visible_devices is not None and torch.cuda.is_available():
        print(f"Using GPU: {visible_devices}")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    
    return device

class MetaIRL(nn.Module):
    """Class that implements DemoDICE training in PyTorch"""
    def __init__(self, state_dim, action_dim, is_discrete_action: bool, config):
        super(MetaIRL, self).__init__()
        self.device = get_device()
        hidden_size = config['hidden_size']
        critic_lr = config['critic_lr']
        actor_lr = config['actor_lr']
        self.is_discrete_action = is_discrete_action
        self.grad_reg_coeffs = config['grad_reg_coeffs']
        self.discount = config['gamma']
        self.non_expert_regularization = config['alpha'] + 1.

        # 定义网络
        self.cost = utils.Critic(state_dim, action_dim, hidden_size=hidden_size,
                                 use_last_layer_bias=config['use_last_layer_bias_cost'],
                                 kernel_initializer=config['kernel_initializer']).to(self.device)
        self.critic = utils.Critic(state_dim, 0, hidden_size=hidden_size,
                                   use_last_layer_bias=config['use_last_layer_bias_critic'],
                                   kernel_initializer=config['kernel_initializer']).to(self.device)
        if self.is_discrete_action:
            self.actor = utils.DiscreteActor(state_dim, action_dim).to(self.device)
        else:
            self.actor = utils.MetaTanhActor(state_dim, action_dim, hidden_size=hidden_size).to(self.device)

        # 定义优化器
        self.cost_optimizer = optim.Adam(self.cost.parameters(), lr=critic_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr = actor_lr
    def update(self, init_states, expert_states, expert_actions, expert_next_states, expert_dones,
               union_states, union_actions, union_next_states, union_dones):
        # 将输入转换为 PyTorch 张量
        init_states = torch.tensor(init_states, dtype=torch.float32).to(self.device)
        expert_states = torch.tensor(expert_states, dtype=torch.float32).to(self.device)
        expert_actions = torch.tensor(expert_actions, dtype=torch.float32).to(self.device)
        expert_next_states = torch.tensor(expert_next_states, dtype=torch.float32).to(self.device)
        expert_dones = torch.tensor(expert_dones, dtype=torch.float32).to(self.device)
        union_states = torch.tensor(union_states, dtype=torch.float32).to(self.device)
        union_actions = torch.tensor(union_actions, dtype=torch.float32).to(self.device)
        union_next_states = torch.tensor(union_next_states, dtype=torch.float32).to(self.device)
        union_dones = torch.tensor(union_dones, dtype=torch.float32).to(self.device)

        # 定义输入
        expert_inputs = torch.cat([expert_states, expert_actions], dim=-1).to(self.device)
        union_inputs = torch.cat([union_states, union_actions], dim=-1).to(self.device)

        # 计算成本函数的输出
        expert_cost_val = self.cost(expert_inputs)
        union_cost_val = self.cost(union_inputs)

        union_cost = torch.sigmoid(union_cost_val)
        expert_cost = torch.sigmoid(expert_cost_val)
        # nu learning
        expert_nu = self.critic(expert_states)
        expert_next_nu = self.critic(expert_next_states)
        union_nu = self.critic(union_states)
        union_next_nu = self.critic(union_next_states)

        union_target_qs =  union_cost.detach() + (1.0 - union_dones) * self.discount * union_next_nu 
        expert_target_qs =  expert_cost.detach() + (1.0 - expert_dones) * self.discount * expert_next_nu 
        union_loss =  (union_target_qs - union_nu).pow_(2).mean()
        expert_loss = (expert_target_qs - expert_nu).pow_(2).mean()
        nu_loss = union_loss + expert_loss
        union_target_qs = union_cost + (1.0 - union_dones) * self.discount * union_next_nu  #core of meta
        # weighted BC
        weight = union_target_qs
        log_probs = self.actor.get_log_prob(union_states, union_actions)
        pi_loss = - (weight * log_probs).mean() #core of meta
        fast_weights = self.actor.net.parameters()
        grad = torch.autograd.grad(pi_loss, fast_weights, create_graph=True,retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.actor_lr * p[0], zip(grad, fast_weights)))
        meta_log_probs = self.actor.get_log_prob(expert_states, expert_actions, fast_weights)
        meta_loss =  - meta_log_probs.mean()

        # 清零梯度
        self.cost_optimizer.zero_grad()
        cost_final_loss =  meta_loss 
        cost_final_loss.backward(retain_graph=True)
        self.cost_optimizer.step()
        pi_loss = - (weight.detach() * log_probs).mean()
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        # 计算梯度并更新参数
        nu_loss.backward(retain_graph=True)
        pi_loss.backward()

        self.critic_optimizer.step()
        self.actor_optimizer.step()

        info_dict = {
            "meta_loss": meta_loss.item(), 
            "cost_final_loss": cost_final_loss.item(),
            'nu_loss': nu_loss.item(),
            'actor_loss': pi_loss.item(),
            'expert_nu': expert_nu.mean().item(),
            'union_nu': union_nu.mean().item(),
            'union_adv': union_target_qs.mean().item(),
        }

        return info_dict

    def minimax_discriminator_loss(self, expert_cost_val, union_cost_val):
        # 定义判别器的 Minimax 损失
        real_labels = torch.ones_like(expert_cost_val)
        fake_labels = torch.zeros_like(union_cost_val)

        loss_fn = nn.BCEWithLogitsLoss()

        loss_real = loss_fn(expert_cost_val, real_labels)
        loss_fake = loss_fn(union_cost_val, fake_labels)

        total_loss = loss_real + loss_fake

        return total_loss

    def step(self, observation, deterministic: bool = True):
        observation = torch.tensor([observation], dtype=torch.float32).to(self.device)
        all_actions, _ = self.actor(observation)
        if deterministic:
            action = all_actions[0]
        else:
            action = all_actions[1]
        return action.detach().cpu().numpy()[0]

    def get_training_state(self):
        training_state = {
            'cost_state_dict': self.cost.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'cost_optimizer_state_dict': self.cost_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
        }
        return training_state

    def set_training_state(self, training_state):
        self.cost.load_state_dict(training_state['cost_state_dict'])
        self.critic.load_state_dict(training_state['critic_state_dict'])
        self.actor.load_state_dict(training_state['actor_state_dict'])
        self.cost_optimizer.load_state_dict(training_state['cost_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(training_state['critic_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(training_state['actor_optimizer_state_dict'])

    def init_dummy(self, state_dim, action_dim):
        # dummy train_step (to create optimizer variables)
        dummy_state = np.zeros((1, state_dim), dtype=np.float32)
        dummy_action = np.zeros((1, action_dim), dtype=np.float32)
        self.update(dummy_state, dummy_state, dummy_action, dummy_state, dummy_state, dummy_action, dummy_state)

    def save(self, filepath, training_info):
        print('Save checkpoint: ', filepath)
        training_state = self.get_training_state()
        data = {
            'training_state': training_state,
            'training_info': training_info,
        }
        with open(filepath + '.tmp', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(filepath + '.tmp', filepath)
        print('Saved!')

    def load(self, filepath):
        print('Load checkpoint:', filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.set_training_state(data['training_state'])
        return data

