import numpy as np
import random
from random import randint

class TreasureEnv():
    def __init__(self, size: int = 3, seed: int = 1337):
        """
        Init (i, j) world
        """
        random.seed(seed)

        self.size = size

        self.__target_location = np.array([randint(0, self.size-1), randint(0, self.size-1)])
        self.agent_location  = np.array([-1, -1])

        self.action_space = 4
        self.action_to_direction = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]), # Down
            2: np.array([0, -1]), # Left
            3: np.array([0, 1]),  # Right
        }

    def reset(self):
        """
        Returns
            observation (agent and target location) and distance info
        """
        self.agent_location  = np.array([randint(0, self.size-1), randint(0, self.size-1)])
        
        while np.array_equal(self.agent_location, self.__target_location):
            self.agent_location = np.array([randint(0, self.size-1), randint(0, self.size-1)])

        obs  = self.__get_obs()
        info = self.__get_info()

        return obs, info
    
    def step(self, action):
        """
        Returns
            obs, reward, terminated, info
        """
        self.agent_location += self.action_to_direction[action]

        obs = self.__get_obs()
        info = self.__get_info()
        
        terminated = np.array_equal(self.agent_location, self.__target_location)
        reward = 1 if terminated else 0

        return obs, reward, terminated, info
    
    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                if np.array_equal([i, j], self.agent_location):
                    print('[O]', end='')
                elif np.array_equal([i, j], self.__target_location):
                    print('[X]', end='')
                else:
                    print('[ ]', end='')
            print()

    def __get_obs(self):
        return {"agent": self.agent_location.copy(), "target": self.__target_location.copy()}
    
    def __get_info(self):
        return {"distance": self.__calc_distance()}

    def __calc_distance(self):
        """
        Returns
            Manhattan distance
        """
        return abs(self.agent_location[0] - self.__target_location[0]) + abs(self.agent_location[1] - self.__target_location[1])




