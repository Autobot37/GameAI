import torch
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
        self.layer1 = nn.Linear(32, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(1, -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        pass


class QAgent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN(4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    def get_state(self, game):
        pass
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
        if self.n_games < 20:
            final_move = random.randint(0,4)
        else:
            state = torch.tensor(state, dtype=torch.float)
            pred = self.model(state)
            move = torch.argmax(pred).item()
            final_move = move
        return move


def train():
    scores = []
    total_score = 0
    max_score = 0
    agent = QAgent() 
    game = Game()     
    while True:
        state = agent.get_state(game)
        action = agent.get_action(state)

        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        agent.train_single(state, action, reward, state_new, done)
        agent.push(state, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_buffer()

            if score > max_score:
                max_score = score
                #save model
            
            print(f"Game:{agent.n_games},Score:{score}, Max Score:{max_score}")
            scores.append(score)
            


