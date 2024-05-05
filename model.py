import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from game import *
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_MEMORY = 10_000
BATCH_SIZE = 1024
LR = 1e-4

class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.layer1 = nn.Linear(984064, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)[0]

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        state = state.permute(2, 0, 1)
        tf = torchvision.transforms.Resize((128, 128))
        state = tf(state)
        next_state = next_state.permute(2, 0, 1)
        next_state = tf(next_state)

        pred_q = self.model(state)
        with torch.no_grad():
            target_q = pred_q.clone()
            target_q[action] = reward + (1 - done) * self.gamma * torch.max(self.model(next_state))
        loss = self.criterion(pred_q, target_q)
        self.optimizer.zero_grad()
        self.optimizer.step()  

class QAgent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN(4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
  
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train_single(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    def train_buffer(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory
        
        for state, action, reward, next_state, done in sample:
            self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = 0
        if self.n_games < 5:
            final_move = random.randint(0,3)
        else:
            state = torch.tensor(state, dtype=torch.float)
            state = state.permute(2, 0, 1)
            tf = torchvision.transforms.Resize((128, 128))
            state = tf(state)
            pred = self.model(state)
            move = torch.argmax(pred).item()
            final_move = move
        return final_move

dir_dict = {
    0:"UP",
    1:"DOWN",
    2:"LEFT",
    3:"RIGHT"
}

def train():
    scores = []
    max_score = 0
    agent = QAgent() 
    game = Game()     
    while True:
        state = game.get_state()
        action = agent.get_action(state)
        reward, done = game.run_agent(dir_dict[action])
        state_new = game.get_state()
        assert state.shape == state_new.shape
        agent.train_single(state, action, reward, state_new, done)
        agent.push(state, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_buffer()

            if reward > max_score:
                max_score = reward
                #save model
            
            print(f"Game:{agent.n_games},Current Score:{reward}, Max Score:{max_score}")
            scores.append(reward)


if __name__ == "__main__":
    train()