import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from .replay_buffers import BasicBuffer
from .dqn_model import ConvDQN, DQN


class DQNAgent:

    def __init__(self,  use_conv=False, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(6,5).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.00001)
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.40):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals1, qvals2 = self.model.forward(state)
        action1 = np.argmax(qvals1.cpu().detach().numpy())
        action2 = np.argmax(qvals2.cpu().detach().numpy())
        if(np.random.randn() < eps):
            action1=np.random.randint(0, 4)
        if(np.random.randn() < eps):
            action2=np.random.randint(0, 4)

        return action1, action2

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = np.array(actions)

        actions1, actions2 = actions[:, 0], actions[:, 1]
        actions1 = torch.LongTensor(actions1).to(self.device)
        actions2 = torch.LongTensor(actions2).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q1, curr_Q2 = self.model.forward(states)
        curr_Q1 = curr_Q1.gather(1, actions1.unsqueeze(1)).squeeze(1)
        curr_Q2 = curr_Q2.gather(1, actions2.unsqueeze(1)).squeeze(1)

        next_Q1, next_Q2 = self.model.forward(next_states)
        max_next_Q1 = torch.max(next_Q1, 1)[0]
        max_next_Q2 = torch.max(next_Q2, 1)[0]

        expected_Q1 = rewards.squeeze(1) + self.gamma * max_next_Q1
        expected_Q2 = rewards.squeeze(1) + self.gamma * max_next_Q2

        loss1 = self.MSE_loss(curr_Q1, expected_Q1)
        loss2 = self.MSE_loss(curr_Q2, expected_Q2)

        loss = loss1 + loss2
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
