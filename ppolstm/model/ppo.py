import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

class Actor(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, 
                 fc1_dims=64, fc2_dims=64, chkpt_dir='model/tmp'):
        super(Actor, self).__init__()
        self.checkpoint_file = 'actor_torch_ppo' #os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        return Categorical(self.actor(state))
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Critic(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=64, fc2_dims=64,  chkpt_dir='tmp/ppo'):
        super(Critic, self).__init__()
        self.checkpoint_file = 'actor_torch_ppo' #os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        return self.critic(state)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size
    
    def generate_batches(self):
        n_states = len(self.states) - 1
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size].item() for i in batch_start]
        return batches

    def load_states(self):
        return self.states
    
    def load_memory(self):
        return torch.stack(self.states).to(device),\
                torch.tensor(self.actions).to(device),\
                torch.tensor(self.probs).to(device),\
                torch.tensor(self.values).to(device),\
                torch.tensor(self.rewards).to(device),\
                torch.tensor(self.dones).to(device)
                  
    def store_memory(self, state, action, probs, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

class PPO(nn.Module):
    def __init__(self, n_actions, input_dims, hidden_size, lstm_num_layers=2,
                 gamma=0.999, lamb=0.95, alpha=0.003, policy_clip=0.2, 
                 batch_size=64):
        super(PPO, self).__init__()
        self.lstm = nn.LSTM(input_dims, hidden_size, num_layers=lstm_num_layers, batch_first=True, device=device)
        self.lstm_optimizer = optim.Adam(self.lstm.parameters(), lr=alpha)

        self.actor = Actor(n_actions, hidden_size, alpha)
        self.critic = Critic(hidden_size, alpha)

        self.memory = PPOMemory(batch_size)
        self.gamma = gamma
        self.lamb = lamb
        self.policy_clip = policy_clip
    
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def select_action(self, state):
        states = self.memory.load_states()
        states = torch.stack([*states, state])

        output, (h, c) = self.lstm(states)
        dist: Categorical = self.actor(output[-1])
        value = self.critic(output[-1])
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    
    def learn(self, n_epoch=10):
        states, actions, old_probs, old_values, rewards, dones = self.memory.load_memory()
        advantages = self.calculate_advantage(rewards, old_values, dones)

        for _ in range(n_epoch):
            batches = self.memory.generate_batches()
            for batch in batches:
                output, (h, c) = self.lstm(states[:batch+1, :])

                dist: Categorical = self.actor(output[0])
                value = self.critic(output[0]).squeeze()
                
                new_probs = dist.log_prob(actions[batch])
                entropy = dist.entropy()

                prob_ratio = new_probs.exp() / old_probs[batch].exp()
                w_probs = advantages[batch] * prob_ratio
                w_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantages[batch]

                actor_loss = -torch.min(w_probs, w_clipped_probs).mean()
                
                returns = (advantages[batch] + old_values[batch]).squeeze()
                critic_loss = F.mse_loss(value, returns)

                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                self.lstm.zero_grad()

                total_loss.backward()

                self.lstm_optimizer.step()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def calculate_advantage(self, rewards, values, dones):
        deltas = rewards[:-1] + self.gamma * values[1:] * (~dones[:-1]) - values[:-1]
        discount_factors = torch.pow(self.gamma * self.lamb, torch.arange(len(deltas), dtype=torch.float32))
        discount_factors = discount_factors.view(1, 1, -1).to(device)
        deltas = deltas.view(1, 1, -1)
        advantages = torch.zeros_like(deltas)
        for t in range(deltas.size(2)):
            advantages[:, :, t] = torch.sum(deltas[:, :, t:] * discount_factors[:, :, :deltas.size(2)-t], dim=2)
        return advantages.view(-1)