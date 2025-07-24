import numpy as np
from matplotlib import pyplot as plt

def check_response(env, obs, reward):
    print(f"agent:  {obs['agent']}")
    print(f"target: {obs['target']}")
    print(f"reward: {reward}")
    env.render()
    print()

def check_q_values(env, agent):
    print("====== Q-values ======")
    for obs in sorted(agent.q_values):
        if np.max(agent.q_values[obs]) > 0:
            for i in range(env.size):
                for j in range(env.size):
                    if (i, j) == (obs[0], obs[1]):
                        print("[O]", end='')
                    elif (i, j) == (obs[2], obs[3]):
                        print("[X]", end='')
                    else:
                        print("[ ]", end='') 
                print()

            print(f"Up: {agent.q_values[obs][0]}")
            print(f"Down: {agent.q_values[obs][1]}")
            print(f"Left: {agent.q_values[obs][2]}")
            print(f"Right: {agent.q_values[obs][3]}")
        print()

def get_moving_average(arr, window, convolution_mode):
    """
    Compute moving average to smooth noisy data
    """
    return np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode)/window

def plot_training_results(arr, size, n_episodes, rolling_length = 500):
    if len(arr) < rolling_length:
        print("len(arr) must >= rolling_length")
        return

    step_moving_average = get_moving_average(arr, rolling_length, "valid")

    plt.figure(figsize=(12, 5))
    plt.title("Episode steps")
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Step")

    plt.plot(range(len(step_moving_average)), step_moving_average)
    plt.savefig(f"experiments/training_results_{size}_{n_episodes}.png")