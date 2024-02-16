import os
import torch
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple


# ---------------------------------------------------------------------------- #
#                                 define model                                 #
# ---------------------------------------------------------------------------- #


class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        # set the random seed manually for comparisons of different changes
        self.seed = torch.manual_seed(seed)
        # model has one 64 node hidden layer
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        # relu is used as activation function
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


# ---------------------------------------------------------------------------- #
#                               make environment                               #
# ---------------------------------------------------------------------------- #

env = gym.make("LunarLander-v2")
# env = gym.make("LunarLander-v2",render_mode="human")
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print("State shape: ", state_shape)
print("State size: ", state_size)
print("Number of actions: ", number_actions)


# ---------------------------------------------------------------------------- #
#                                  parameters                                  #
# ---------------------------------------------------------------------------- #

# learning_rate = 5e-4
learning_rate = 1e-3
min_buffer_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3
update_interval = 4
sample_size = 100

# ---------------------------------------------------------------------------- #
#                             define replay buffer                             #
# ---------------------------------------------------------------------------- #


class ReplayMemory:
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = (
            torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None]))
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)
            )
            .float()
            .to(self.device)
        )
        return states, next_states, actions, rewards, dones


# ---------------------------------------------------------------------------- #
#                              define agent class                              #
# ---------------------------------------------------------------------------- #


class Agent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        # ------------------------------- update model ------------------------------- #
        self.t_step = (self.t_step + 1) % update_interval
        if self.t_step == 0:
            if len(self.memory.memory) > min_buffer_size:
                experiences = self.memory.sample(sample_size)
                self.learn(experiences, discount_factor)

    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        # state = torch.from_numpy(state).float().to(self.device)
        # state = torch.from_numpy(state).to(self.device) # 757,201.06

        # set the model to evaluation mode which grid will not be recorded
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        # set the model back to train mode
        self.local_qnetwork.train()
        # take action by epsilon-greedy
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = (
            self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        )
        #
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.soft_update(
        #     self.local_qnetwork, self.target_qnetwork, interpolation_parameter
        # )
        self.hard_update(self.local_qnetwork, self.target_qnetwork)

    def hard_update(self, local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())

    def soft_update(self, local_model, target_model, interpolation_parameter):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                interpolation_parameter * local_param.data
                + (1.0 - interpolation_parameter) * target_param.data
            )


graph_score = []
graph_x = []
a = 0

# ---------------------------------------------------------------------------- #
#                                 create agent                                 #
# ---------------------------------------------------------------------------- #
agent = Agent(state_size, number_actions)

# ---------------------------------------------------------------------------- #
#                                     train                                    #
# ---------------------------------------------------------------------------- #

train_episodes = 1400
max_episode_step = 1000
min_epsilon = 0.01
# epsilon_decay = 0.995
epsilon_decay = 0.985
# a = epsilon_decay
epsilon = 1
scores = deque(maxlen=100)

for episode in range(1, train_episodes + 1):
    state, _ = env.reset()
    score = 0
    # prevent stucking in one episode
    for t in range(max_episode_step):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores.append(score)
    epsilon = max(min_epsilon, epsilon_decay * epsilon)

    # graph_score.append(score)
    # graph_x.append(episode)

    print(
        "\rEpisode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores)),
        end="",
    )
    if episode % 100 == 0:
        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                episode, np.mean(scores)
            )
        )
        torch.save(agent.local_qnetwork.state_dict(), f"checkpoint{episode}.pth")
    # if np.mean(scores) >= 200.0:
    #     print(
    #         "\rEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
    #             episode - 100, np.mean(scores)
    #         )
    #     )
    #     graph_x.append(episode - 100)
    #     graph_score.append(np.mean(scores))
    #     break

torch.save(agent.local_qnetwork.state_dict(), f"finalcheckpoint.pth")
# print(
#     "\rEnnnnnnnnnt solved in {:d} episodes!\tAverage Score: {:.2f}".format(
#         np.mean(graph_x), np.mean(graph_score)
#     )
# )
# print(a)

# plt.plot(graph_x,graph_score)
# plt.show()
# agent = Agent(state_size, number_actions)

# agent.local_qnetwork.load_state_dict(torch.load("checkpoint_best.pth"))

# state, info = env.reset()
# # env.render()
# for _ in range(5):
#     state, info = env.reset()
#     for i in range(900):
#         action = agent.act(state,-1)
#         next_state, reward, done, _, _ = env.step(action)
#         if done: break
#         state = next_state
