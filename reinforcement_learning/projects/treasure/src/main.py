import os
from dotenv import load_dotenv

from treasure_env   import TreasureEnv
from treasure_agent import TreasureAgent

load_dotenv("config/.env")
size = int(os.getenv("SIZE"))
n_episodes = int(os.getenv("N_EPISODES"))

env   = TreasureEnv(size=size)
agent = TreasureAgent(env)

# Training
for episode in range(n_episodes):
    obs, info = env.reset()
    print(f"========= episode {episode} =========")
    print(f"agent:  {obs['agent']}")
    print(f"target: {obs['target']}")
    env.render()
    print()

    terminated = False
    while not terminated:
        action = agent.get_action()
        obs, reward, terminated, info = env.step(action)

        print(f"agent:  {obs['agent']}")
        print(f"target: {obs['target']}")
        print(f"reward: {reward}")
        env.render()
        print()

# Testing
