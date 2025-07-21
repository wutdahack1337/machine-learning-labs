from random import randint

class TreasureAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self):
        action = randint(0, self.env.action_space-1)
        while not self._check_inside(self.env._agent_location + self.env.action_to_direction[action]):
            action = randint(0, self.env.action_space-1)

        return action
    
    def _check_inside(self, location):
        return (0 <= location[0] and location[0] < self.env.size and 0 <= location[1] and location[1] < self.env.size)