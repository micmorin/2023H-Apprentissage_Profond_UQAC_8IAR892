import torch
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym


from gymnasium import spaces
from os.path import isfile
from Game.Board import Board
from Game.Levels import Levels
from Game.User import User
from CNN import cnn


class dqn():   
    def __init__(self, cnn, board, user):
        self.cnn = cnn
        self.board = board
        self.user = user
        self.moves_left = self.board.level.nbr_moves
        self.train()

    def train(self):
        # TAKEN FROM pytorch.org/tutorials/intermediate/reinforcement_q_learning.html, with modifications
        EPISODES = 1000
        LEARNING_RATE = 0.0001
        MEM_SIZE = 10000
        BATCH_SIZE = 64
        GAMMA = 0.95
        EXPLORATION_MAX = 1.0
        EXPLORATION_DECAY = 0.999
        EXPLORATION_MIN = 0.001

        FC1_DIMS = 1024
        FC2_DIMS = 512
        DEVICE = torch.device("cpu")

        env = CustomEnv(self.cnn,self.board,self.user)
        observation_space = env.observation_space.shape[1]
        action_space = env.action_space.n

        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_shape = env.observation_space.shape[1]
                self.action_space = action_space

                self.fc1 = nn.Linear(self.input_shape, FC1_DIMS)
                self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
                self.fc3 = nn.Linear(FC2_DIMS, self.action_space)

                self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
                self.loss = nn.MSELoss()
                self.to(DEVICE)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)

                return x

        class ReplayBuffer:
            def __init__(self):
                self.mem_count = 0
                
                self.states = np.zeros((MEM_SIZE, env.observation_space.shape[1]),dtype=np.float32)
                self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
                self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
                self.states_ = np.zeros((MEM_SIZE, env.observation_space.shape[1]),dtype=np.float32)
                self.dones = np.zeros(MEM_SIZE, dtype=np.bool_)
            
            def add(self, state, action, reward, state_, done):
                mem_index = self.mem_count % MEM_SIZE
                
                self.states[mem_index]  = state
                self.actions[mem_index] = action
                self.rewards[mem_index] = reward
                self.states_[mem_index] = state_
                self.dones[mem_index] =  1 - done

                self.mem_count += 1
            
            def sample(self):
                MEM_MAX = min(self.mem_count, MEM_SIZE)
                batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
                
                states  = self.states[batch_indices]
                actions = self.actions[batch_indices]
                rewards = self.rewards[batch_indices]
                states_ = self.states_[batch_indices]
                dones   = self.dones[batch_indices]

                return states, actions, rewards, states_, dones
        
        class DQN_Solver:
                        def __init__(self):
                            self.memory = ReplayBuffer()
                            self.exploration_rate = EXPLORATION_MAX
                            self.network = Network()

                        def choose_action(self, observation):
                            if random.random() < self.exploration_rate:
                                return env.action_space.sample()
                            
                            state = torch.tensor(observation).float().detach()
                            state = state.to(DEVICE)
                            state = state.unsqueeze(0)
                            q_values = self.network(state)
                            return torch.argmax(q_values).item()
                        
                        def learn(self):
                            if self.memory.mem_count < BATCH_SIZE:
                                return
                            
                            states, actions, rewards, states_, dones = self.memory.sample()
                            states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
                            actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
                            rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
                            states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
                            dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
                            batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

                            q_values = self.network(states)
                            next_q_values = self.network(states_)
                            
                            predicted_value_of_now = q_values[batch_indices, actions]
                            predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
                            
                            q_target = rewards + GAMMA * predicted_value_of_future * dones

                            loss = self.network.loss(q_target, predicted_value_of_now)
                            self.network.optimizer.zero_grad()
                            loss.backward()
                            self.network.optimizer.step()

                            self.exploration_rate *= EXPLORATION_DECAY
                            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

                        def returning_epsilon(self):
                            return self.exploration_rate

        if isfile('model.pth'):
            self.agent = DQN_Solver()
            self.agent.network.load_state_dict(torch.load('model.pth'))
            self.agent.network.eval()
        else:
            best_reward = 0
            average_reward = 0
            episode_number = []
            average_reward_number = []

            self.agent = DQN_Solver()
            self.agent.network.load_state_dict(torch.load('model.pth'))

            for i in range(1, EPISODES):
                state = env.reset()
                state = np.reshape(state, (1, observation_space))
                score = 0

                while True:
                    #env.render()
                    action = self.agent.choose_action(state)
                    state_, reward, done, info = env.step(action)
                    state_ = np.reshape(state_, (1, observation_space))
                    self.agent.memory.add(state, action, reward, state_, done)
                    self.agent.learn()
                    state = state_
                    score += reward

                    if done:
                        if score > best_reward:
                            best_reward = score
                        average_reward += score 
                        print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, self.agent.returning_epsilon()))
                        break
                        
                    episode_number.append(i)
                    average_reward_number.append(score)

            torch.save(self.agent.network.state_dict(), 'model.pth')
            plt.plot(episode_number, average_reward_number)
            plt.show()
        

    def solve(self, cnn, board, user):
        env = CustomEnv(cnn,board,user)
        count = 0
        while True:
            move = self.agent.choose_action(env._get_obs())
            move = int(move)
            # 0000 = 0, 0010 = 6, 0100 = 36, 1000 = 216, 5555 = 1295
            move = str(math.floor(move/216))+str(math.floor(move%216/36))+str(math.floor(move%216%36/6))+str(math.floor(move%216%36%6))
            move = user.getMove(move)
            count = count+1
            if board.verifyMove(move):
                return move, count

class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, cnn, board, user, render_mode=None):
        self.cnn = cnn
        self.board = board
        self.user = user
        self.moves_left = self.board.level.nbr_moves

        # We have 3 matrixes of 6x6
        self.observation_space = spaces.Box(1, 60, shape=(1,108), dtype=int)

        # We have actions as "x1y1x2y2"
        self.action_space = spaces.Discrete(1296)

    def _get_obs(self):
        self.cnn.visualMatrixToMatrixForAgent(self.board, self.user)
        powerGrid = []
        team = self.user.getPower()
        for row in self.cnn.matrix:
            temp = []
            for value in row:
                try:
                    temp.append(team[value-1])
                except:
                    temp.append(0)
            powerGrid.append(temp)

            # [pkmn_grid,barrier_grid,power_grid]
        return np.reshape(np.array([self.cnn.matrix, self.board.barriers, powerGrid], dtype=int),(1,108)),

    def reset(self):
        self.board.initialMatrix(self.board.level)
        self.moves_left = self.board.level.nbr_moves
        return self._get_obs()
    
    def step(self, action):
        terminated = False
        action = int(action)
        # 0000 = 0, 0010 = 6, 0100 = 36, 1000 = 216, 5555 = 1295
        action = str(math.floor(action/216))+str(math.floor(action%216/36))+str(math.floor(action%216%36/6))+str(math.floor(action%216%36%6))
        move = self.user.getMove(action)
        reward = 0
        

        if self.board.verifyMove(move):
            self.board.enactMove(move, self.board.level, self.user)
            self.moves_left -= 1

            if self.board.level.hp < self.board.level.score: 
                terminated = True
                reward = self.moves_left

            if self.moves_left == 0: 
                terminated = True
                reward = -4

        # If move is not valid
        else:
            if self.cnn.matrix[int(move[0]),int(move[1])] == self.board.layout[int(move[0]),int(move[1])] \
              and self.cnn.matrix[int(move[2]),int(move[3])] == self.board.layout[int(move[2]),int(move[3])]:
                reward = -1
        
        observation = self._get_obs()
        return observation, reward, terminated, ""