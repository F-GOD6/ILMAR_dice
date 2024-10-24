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

class MetaISWBC(nn.Module):
    """Class that implements DemoDICE training in PyTorch"""
    def __init__(self, state_dim, action_dim, is_discrete_action: bool, config):
        super(MetaISWBC, self).__init__()
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
        if self.is_discrete_action:
            self.actor = utils.DiscreteActor(state_dim, action_dim).to(self.device)
        else:
            self.actor = utils.MetaTanhActor(state_dim, action_dim, hidden_size=hidden_size).to(self.device)

        # 定义优化器
        self.cost_optimizer = optim.Adam(self.cost.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr = actor_lr
        self.alpha = config['walpha']
        self.beta = config['beta']
        self.tau = config['tau']
    def update(self, init_states, expert_states, expert_actions, expert_next_states,
               union_states, union_actions, union_next_states):
        # 将输入转换为 PyTorch 张量
        init_states = torch.tensor(init_states, dtype=torch.float32).to(self.device)
        expert_states = torch.tensor(expert_states, dtype=torch.float32).to(self.device)
        expert_actions = torch.tensor(expert_actions, dtype=torch.float32).to(self.device)
        expert_next_states = torch.tensor(expert_next_states, dtype=torch.float32).to(self.device)
        union_states = torch.tensor(union_states, dtype=torch.float32).to(self.device)
        union_actions = torch.tensor(union_actions, dtype=torch.float32).to(self.device)
        union_next_states = torch.tensor(union_next_states, dtype=torch.float32).to(self.device)

        # 定义输入
        expert_inputs = torch.cat([expert_states, expert_actions], dim=-1).to(self.device)
        union_inputs = torch.cat([union_states, union_actions], dim=-1).to(self.device)

        # 计算成本函数的输出
        expert_cost_val = self.cost(expert_inputs)
        union_cost_val = self.cost(union_inputs)

        # 创建混合输入用于梯度惩罚
        batch_size = expert_states.size(0)
        unif_rand = torch.rand(batch_size, 1)
        unif_rand = unif_rand.to(expert_states.device)

        mixed_inputs1 = unif_rand * expert_inputs + (1 - unif_rand) * union_inputs
        mixed_inputs2 = unif_rand * union_inputs[torch.randperm(batch_size)] + (1 - unif_rand) * union_inputs
        mixed_inputs = torch.cat([mixed_inputs1, mixed_inputs2], dim=0)
        mixed_inputs.requires_grad_(True)

        # 梯度惩罚 for cost
        cost_output = self.cost(mixed_inputs)
        cost_output = torch.log(1 / (torch.sigmoid(cost_output) + EPS2) - 1 + EPS2)

        cost_grad = torch.autograd.grad(
            outputs=cost_output,
            inputs=mixed_inputs,
            grad_outputs=torch.ones_like(cost_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0] + EPS

        cost_grad_penalty = ((cost_grad.norm(2, dim=-1) - 1) ** 2).mean()

        cost_loss = self.minimax_discriminator_loss(expert_cost_val, union_cost_val) \
                    + self.grad_reg_coeffs[0] * cost_grad_penalty

        cost_prob = torch.nn.Sigmoid()(union_cost_val)
        weight = (cost_prob / (1 - cost_prob))
        indices = (weight >= self.tau).float()
        log_probs = self.actor.get_log_prob(union_states, union_actions)
        pi_loss = - (indices * weight * log_probs).mean() #core of meta
        fast_weights = self.actor.net.parameters()
        grad = torch.autograd.grad(pi_loss, fast_weights, create_graph=True,retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.actor_lr * p[0], zip(grad, fast_weights)))
        meta_log_probs = self.actor.get_log_prob(expert_states, expert_actions, fast_weights)
        meta_loss =  - meta_log_probs.mean()
        # 清零梯度
        self.cost_optimizer.zero_grad()
        cost_final_loss = self.alpha * meta_loss + self.beta * cost_loss
        cost_final_loss.backward(retain_graph=True)
        self.cost_optimizer.step()
        pi_loss = - (indices * weight.detach() * log_probs).mean()
        self.actor_optimizer.zero_grad()
        # 计算梯度并更新参数
        pi_loss.backward()
        self.actor_optimizer.step()

        info_dict = {
            'cost_loss': cost_loss.item(),
            "meta_loss": meta_loss.item(), 
            'cost_final_loss': cost_final_loss.item(),
            'actor_loss': pi_loss.item(),
            'expert_cost_val': expert_cost_val.mean().item(),
            'union_cost_val': union_cost_val.mean().item(),   
            'indices': indices.mean().item(),
            'weight' : weight.mean().item()      
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
            'actor_state_dict': self.actor.state_dict(),
            'cost_optimizer_state_dict': self.cost_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
        }
        return training_state

    def set_training_state(self, training_state):
        self.cost.load_state_dict(training_state['cost_state_dict'])
        self.actor.load_state_dict(training_state['actor_state_dict'])
        self.cost_optimizer.load_state_dict(training_state['cost_optimizer_state_dict'])
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

