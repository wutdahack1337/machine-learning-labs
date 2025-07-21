import datetime
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from matplotlib import pyplot as plt

from blackjack import BlackjackAgent
from utils import get_moving_avgs

# Training hyperparameters
learning_rate  = 0.06    # How fast to learn (higher = faster but less stable)
n_episodes     = 333_333 # Number of hands to practice 
start_epsilon  = 1.0     # Start with 100% random actions   
epsilon_decay  = start_epsilon/(n_episodes/1.5) # Reduce exploration over time
final_epsilon  = 0.1     # Always keep some exploration

# Create enviroment and agent
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(env=env, learning_rate=learning_rate, initial_epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset() 
    done = False

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)
    
        # Learn from this experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Move to next state
        done = terminated or truncated
        obs  = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()

env.close()

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12,5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Average Reward")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Average Episode Length")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(agent.training_error, rolling_length, "same")
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_xlabel("Step")
axs[2].set_ylabel("Temporal Difference Error")

plt.tight_layout()
plt.show()
fig.savefig(f"experiments/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

# Test the trained agent
def test_agent(agent, env, num_episodes = 1000):
    """
    Test agent performance without learning or exploration
    """
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0 # Pure exploitation

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action =  agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} epsildes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")
    plt.hist(total_rewards)
    plt.show()

# Test your agent
test_agent(agent, env)