import numpy as np
import gymnasium as gym
from typing import Optional
from gymnasium.utils.env_checker import check_env

class GridWorldEnv(gym.Env):
    def __init__(self, size: int = 5):
        # The size of the square grid (5x5 by default)
        self.size = size

        # Initialize positions - will be set randomly in reset()
        # Using -1, -1 as "uninitialized" state
        self._agent_location  = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent":  gym.spaces.Box(0, size-1, shape=(2,), dtype=int), # [x, y] coordinates
                "target": gym.spaces.Box(0, size-1, shape=(2,), dtype=int), # [x, y] coordinates
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)
        
        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([1, 0]),  # Move riggt (positive x)
            1: np.array([0, 1]),  # Move up (positive y)
            2: np.array([-1, 0]), # Move left (negative x)
            3: np.array([0, -1]), # Move down (negative y)
        }

    def _get_obs(self):
        """
        Convert internal state to observation format

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        """
        Compute auxiliary information for debugging

        Returns:
            dict: Info with distance between agent and target
        """
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Start a new episode

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:  
            tuple: (observation, info) for the initial state
        """
        
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info        = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one timestep within the enviroment

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Check if agent reached the target
        terminated  = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated   = False

        # Simple reward stucture: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward      = 1 if terminated else 0
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    

# Register the enviroment so we can create it with gym.make()
gym.register(
    id="custom_environments/GridWorld-v0",
    entry_point=GridWorldEnv,
    max_episode_steps=300,
)

env = gym.make("custom_environments/GridWorld-v0")

# This will catch many common issues
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")

# Test specific action sequences to verify behavior
obs, info = env.reset(seed=1337) # Use seed for reproducible testing

print(f"Starting positio - Agent: {obs["agent"]}, Target: {obs["target"]}")

# Test each action type
actions = [0, 1, 2, 3] # right, up, left, down
for action in actions:
    old_pos = obs["agent"].copy()
    obs, reward, terminated, truncated, info = env.step(action)
    new_pos = obs["agent"]
    print(f"Action {action}: {old_pos} -> {new_pos}, reward={reward}")