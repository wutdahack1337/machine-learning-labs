from tqdm import tqdm

import utils
from config  import TrainingConfig
from treasure_env   import TreasureEnv
from treasure_agent import TreasureAgent

def main():
    """
    Treasure hunting agent
    """
    config = TrainingConfig()
    size       = config.size
    seed       = config.seed
    train_episodes  = config.train_episodes
    learning_rate   = config.learning_rate
    discount_factor = config.discount_factor
    epsilon = config.epsilon
    test_episodes = config.test_episodes 

    env   = TreasureEnv(size=size, seed=seed)
    agent = TreasureAgent(env, learning_rate, epsilon, discount_factor, seed)

    # Training
    print(f"Training agent on {size}x{size} world for {train_episodes} episodes...")
    step_cnt_queue = train_agent(env, agent, train_episodes)

    # Analysis
    utils.check_q_values(env, agent)
    utils.plot_training_results(step_cnt_queue, size, train_episodes)

    # Testing
    print("Testing trained agent (exploitation only)...")
    test_agent(env, agent, test_episodes)

def train_agent(env, agent, n_episodes):
    """
    Train the agent
    """
    step_cnt_queue = []
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs, info = env.reset()
        step_cnt = 0
        terminated = False

        while not terminated:
            action = agent.get_action(obs, verbose=0)
            next_obs, reward, terminated, info = env.step(action)
            step_cnt += 1

            agent.learn(obs, action, reward, terminated, next_obs)
            obs = next_obs

        step_cnt_queue.append(step_cnt)

    return step_cnt_queue
    
def test_agent(env, agent, test_episodes):
    """
    Test the trained agent
    """
    original_epsilon = agent.epsilon
    agent.epsilon    = 0.0

    for episode in range(test_episodes):
        obs, info = env.reset()
        print(f"========= Test episode {episode + 1} =========")
        utils.check_response(env, obs, 0)

        terminated = False
        while not terminated:
            action = agent.get_action(obs, verbose=0)
            obs, reward, terminated, info = env.step(action)
            utils.check_response(env, obs, reward)

    agent.epsilon = original_epsilon

if __name__ == "__main__":
    main()