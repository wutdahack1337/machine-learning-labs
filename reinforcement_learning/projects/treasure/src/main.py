import os
from dotenv import load_dotenv
from tqdm import tqdm

import utils
from treasure_env   import TreasureEnv
from treasure_agent import TreasureAgent

load_dotenv("config/.env")
size = int(os.getenv("SIZE"))
n_episodes = int(os.getenv("N_EPISODES"))

if size < 2:
    print("SIZE must >= 2")
    exit(1)

env   = TreasureEnv(size=size, seed=1337)
agent = TreasureAgent(env, seed=1337)

# Training: exploitation and exploration

step_cnt_queue = []
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    # print(f"========= episode {episode} =========")
    # utils.check_response(env, obs, 0)

    step_cnt = 0
    terminated = False
    while not terminated:
        action = agent.get_action(obs, verbose=0)
        next_obs, reward, terminated, info = env.step(action)
        step_cnt += 1

        # utils.check_response(env, obs, reward)

        agent.learn(obs, action, reward, terminated, next_obs)
        obs = next_obs

    step_cnt_queue.append(step_cnt)

# utils.check_q_values(env, agent)
utils.plot_training_results(step_cnt_queue, size, n_episodes)

# Testing: just explotation
