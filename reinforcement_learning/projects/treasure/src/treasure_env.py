from random import randint

class TreasureEnv():
    def __init__(self, size: int = 6):
        """
        Init (i, j) world
        """
        self.size = size

        self._target_location = [-1, -1]
        self._agent_location  = [-1, -1]

        self.action_space = 4
        self.action_to_direction = {
            0: [1, 0],  # Up
            1: [-1, 0], # Down
            2: [0, -1], # Left
            3: [0, 1],  # Right
        }

    def _calc_distance(self):
        """
        Returns
            Mahattan distance
        """
        return abs(self._agent_location[0] - self._target_location[0]) + abs(self._agent_location[1] - self._target_location[1])

    def reset(self):
        """
        Returns
            observation (agent and target location) and distance info
        """
        self._target_location = [randint(0, self.size-1), randint(0, self.size-1)]
        self._agent_location  = [randint(0, self.size-1), randint(0, self.size-1)]
        
        while self._agent_location == self._target_location:
            self._agent_location  = [randint(0, self.size-1), randint(0, self.size-1)]

        return {
            "agent": self._agent_location,
            "target": self._target_location,
        }, self._calc_distance()




