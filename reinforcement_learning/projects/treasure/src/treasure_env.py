import numpy as np
from random import randint

class TreasureEnv():
    def __init__(self, size: int = 3):
        """
        Init (i, j) world
        """
        self.size = size

        self._target_location = np.array([randint(0, self.size-1), randint(0, self.size-1)])
        self._agent_location  = np.array([-1, -1])

        self.action_space = 4
        self.action_to_direction = {
            0: np.array([1, 0]),  # Up
            1: np.array([-1, 0]), # Down
            2: np.array([0, -1]), # Left
            3: np.array([0, 1]),  # Right
        }

    def reset(self):
        """
        Returns
            observation (agent and target location) and distance info
        """
        self._agent_location  = np.array([randint(0, self.size-1), randint(0, self.size-1)])
        
        while np.array_equal(self._agent_location, self._target_location):
            self._agent_location  = np.array([randint(0, self.size-1), randint(0, self.size-1)])

        obs  = self._get_obs()
        info = self._get_info()

        return obs, info
    
    def step(self, action):
        """
        Returns
            obs, reward, terminated, info
        """
        self._agent_location += self.action_to_direction[action]

        obs = self._get_obs()
        info = self._get_info()
        
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = -self._calc_distance()

        return obs, reward, terminated, info 
    
    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                if np.array_equal([i, j], self._agent_location):
                    print('[O]', end='')
                elif np.array_equal([i, j], self._target_location):
                    print('[X]', end='')
                else:
                    print('[ ]', end='')
            print()

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {"distance": self._calc_distance()}

    def _calc_distance(self):
        """
        Returns
            Mahattan distance
        """
        return abs(self._agent_location[0] - self._target_location[0]) + abs(self._agent_location[1] - self._target_location[1])




