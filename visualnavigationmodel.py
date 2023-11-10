# Adapted from https://github.com/jkulhanek/robot-visual-navigation/blob/master/python/model.py#L5

import random
import math
from collections import namedtuple, deque
import configuration

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'last_action', 'last_reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class VisualNavigationNetwork(nn.Module):
    def init_weights(self, module):
        if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

        elif type(module) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
            nn.init.zeros_(module.bias.data)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                module.weight.data)
            d = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(module.weight.data, -d, d)



    def __init__(self, x_resolution, y_resolution, n_actions):
        super(VisualNavigationNetwork, self).__init__()

        self.main_output_size = 512

        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

        self.shared_base = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4),
            nn.ReLU(True),
        )

        self.conv_base = nn.Sequential(
            nn.Conv2d(16, 32, 4, stride=2),  # 9
            nn.ReLU(True),
            nn.Conv2d(32, 32, 1),  # 9
            nn.ReLU(),
        )

        self.conv_merge = nn.Sequential(
            Flatten(),
            nn.Linear(11 * 23 * 32, self.main_output_size),
            nn.ReLU()
        )

        self.critic = nn.Linear(self.main_output_size, 1)
        self.policy_logits = nn.Linear(self.main_output_size, n_actions)

        self.lstm_layers = 1
        self.lstm_hidden_size = self.main_output_size
        self.lstm = nn.LSTM(self.main_output_size + 1 + 1,  # Conv outputs + last action, reward
                                     hidden_size=self.lstm_hidden_size,
                                     num_layers=self.lstm_layers,
                                     batch_first=True)
        
        self.apply(self.init_weights)
        self.pc_cell_size = 2
        self.deconv_cell_size = 2

    def forward(self, image, last_action, last_reward):
        # image is a batch of flat vectors, x * y * color channels
        # reshape flat vectors into batches of 3 planes of x*y each
        image = image.view(-1, self.y_resolution, self.x_resolution, 3)

        # last_action is a batch of single ints, representing which action is taken
        # last_reward is a batch of single ints, representing the previous reward

        # change the ordering so that it is batch, channel, y, x
        image = image.permute(0, 3, 1, 2)
        features = self.shared_base(image)
        features = self.conv_base(features)
        features = self.conv_merge(features)
        features = torch.cat((features, last_action, last_reward), dim=1)
        features, _ = self.lstm(features)
        
        policy_logits = self.policy_logits(features)
        critic = self.critic(features)

        return policy_logits, critic

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 10
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 7e-4
ALPHA = 0.99
EPSILON = 1e-5
ENTROPY_COEFFICIENT = 0.01
VALUE_COEFFICIENT = 0.5
MAX_GRADIENT_NORM = 0.5

"""
        self.max_time_steps = max_time_steps
        self.name = name
        self.num_steps = 5
        self.num_processes = 16
        self.gamma = 0.99
        self.allow_gpu = True
        self.log_dir = None
        self.win = None
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_gradient_norm = 0.5
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5
        self.data_parallel = True
        self.learning_rate = 7e-4
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualNavigationModel():
    def __init__(self, n_actions, network_path=None):
        self.steps_done = 0
        self.n_actions = n_actions
        self.visual_navigation_net = VisualNavigationNetwork(configuration.SCREENSHOT_X_RES, configuration.SCREENSHOT_Y_RES, n_actions).to(device)

        if network_path is not None:
            self.visual_navigation_net.load_state_dict(torch.load(network_path))
        
        
        self.visual_navigation_net.eval()

        self.optimizer = optim.RMSprop(self.visual_navigation_net.parameters(), LR, eps=EPSILON, alpha=ALPHA)
        self.optimizer = optim.AdamW(self.visual_navigation_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def save_network(self, network_path):
        torch.save(self.visual_navigation_net.state_dict(), network_path)

    # get the output of the network, with randomization which decreases over time for training
    def get_result(self, state, last_action, last_reward):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.visual_navigation_net(state, last_action, last_reward)[0].max(1)[1].view(1, 1)
        else:
            return torch.tensor(np.random.randint(0,self.n_actions), device=device, dtype=torch.long)
    
    # get the output of the network with no randomization, for testing purposes only
    def nonrandom_result(self, state, last_action, last_reward):
        return self.visual_navigation_net(state, last_action, last_reward)[0].max(1)[1].view(1, 1)
        
    def save_to_memory(self, *args):
        self.memory.push(*args)

    # Copied from https://github.com/jkulhanek/deep-rl-pytorch/blob/master/deep_rl/actor_critic/a2c.py

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)

        last_action_batch = torch.tensor(batch.action).view(-1,1)
        last_reward_batch = torch.cat(batch.reward)

        # Update learning rate
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = LR

        policy_logits, value = self.visual_navigation_net(state_batch, last_action_batch, last_reward_batch)

        dist = torch.distributions.Categorical(logits=policy_logits)
        action_log_probs = dist.log_prob(last_action_batch)
        dist_entropy = dist.entropy().mean()

        # Compute losses
        advantages = last_reward_batch - value.squeeze(-1)
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()
        loss = value_loss * VALUE_COEFFICIENT + action_loss - dist_entropy * ENTROPY_COEFFICIENT

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.visual_navigation_net.parameters(), MAX_GRADIENT_NORM)
        self.optimizer.step()