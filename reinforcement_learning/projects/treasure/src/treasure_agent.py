import random
import numpy as np
from random import randint
from collections import defaultdict

class TreasureAgent:
    def __init__(self, env, learning_rate: float = 0.01, epsilon: float = 0.1, discount_factor: float = 0.95, seed: int = 1337):
        random.seed(seed)

        self.env = env

        self.epsilon = epsilon

        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space))
        self.learning_rate   = learning_rate
        self.discount_factor = discount_factor

    def get_action(self, obs, verbose = 1):
        obs = tuple(np.concatenate(list(obs.values())))

        if random.random() <= self.epsilon or not np.max(self.q_values[obs]) > 0:
            if verbose == 1:
                print("random action")

            action = randint(0, self.env.action_space-1)
            while not self.__check_inside(self.env.agent_location + self.env.action_to_direction[action]):
                action = randint(0, self.env.action_space-1)
            return action
        else:
            if verbose == 1:
                print("greedy action")
                
            row = self.q_values[obs].copy()
            for action in range (self.env.action_space):
                if not self.__check_inside(self.env.agent_location + self.env.action_to_direction[action]):
                    row[action] = -1
            return int(np.argmax(row)) 
    
    def learn(self, obs, action, reward, terminated, next_obs):
        obs      = tuple(np.concatenate(list(obs.values())))
        next_obs = tuple(np.concatenate(list(next_obs.values())))
    
        future_q_value = (not terminated)*np.max(self.q_values[next_obs])
        target = reward + self.discount_factor*future_q_value
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] = self.q_values[obs][action] + self.learning_rate*temporal_difference

    def __check_inside(self, location):
        return (0 <= location[0] and location[0] < self.env.size and 0 <= location[1] and location[1] < self.env.size)