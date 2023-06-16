import numpy as np
import gym

env = gym.make('CartPole-v1')

# Define the Q-learning parameters
alpha = 0.1     # Learning rate
gamma = 1       # Discount factor
epsilon = 0.2   # Epsilon-greedy parameter

episode_num = 20000

# state space: [postion, d_postion, angle, d_angle]
bin_num = [30, 30, 30, 30]

Q_matrix = np.random.uniform(0, 1, bin_num + [2]) # 2 for action space size

upper_bounds = env.observation_space.high
lower_bounds = env.observation_space.low

# to replace inf value
upper_bounds[1]=3
upper_bounds[3]=10
lower_bounds[1]=-3
lower_bounds[3]=-10


# bin table
cartPosition_bin = np.linspace(lower_bounds[0], upper_bounds[0], bin_num[0])
cartVelocity_bin = np.linspace(lower_bounds[1], upper_bounds[1], bin_num[1])
poleAngle_bin = np.linspace(lower_bounds[2], upper_bounds[2], bin_num[2])
poleAngleVelocity_bin = np.linspace(lower_bounds[3], upper_bounds[3], bin_num[3])


def get_state_index(state):
    # avoiding idx < 0
    cartPosition_bin_idx = np.maximum(np.digitize(state[0], cartPosition_bin) - 1, 0)
    cartVelocity_bin_idx = np.maximum(np.digitize(state[1], cartVelocity_bin) - 1, 0)
    poleAngle_bin_idx = np.maximum(np.digitize(state[2], poleAngle_bin) - 1, 0)
    poleAngleVelocity_bin_idx = np.maximum(np.digitize(state[3], poleAngleVelocity_bin) - 1, 0)
 
    return tuple([cartPosition_bin_idx, cartVelocity_bin_idx, poleAngle_bin_idx, poleAngleVelocity_bin_idx])


def select_action(state, episode_idx):
    global epsilon
    if episode_idx < 100:
        return np.random.choice([0, 1])
    elif episode_idx > 7000:
        epsilon = 0.999 * epsilon

    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice([0, 1])
    else:
        state_idx = get_state_index(state)
        action = np.argmax(Q_matrix[state_idx])
        
    return action

rewards_list = []
for episode_idx in range(episode_num):
    state, _ = env.reset()
    episode_reward = 0
    terminated = False
    while not terminated:
        action = select_action(state, episode_idx)
        state_next, reward, terminated, truncat, _ = env.step(action)
        
        state_idx = get_state_index(state)
        state_next_idx = get_state_index(state_next)
        
        if not (terminated or truncat):  # is not the last state
            Q_matrix[state_idx + (action,)] = (1 - alpha) * Q_matrix[state_idx + (action,)] + alpha * (reward + gamma * np.max(Q_matrix[state_next_idx]))
        else:           # is the last state
            Q_matrix[state_idx + (action,)] = (1 - alpha) * Q_matrix[state_idx + (action,)] + alpha * reward
        
        state = state_next
        episode_reward += reward
        
    print("Episode: {}, reward: {}".format(episode_idx, episode_reward))
    rewards_list.append(episode_reward)

env.close()

# conclusion
import matplotlib.pyplot as plt
plt.plot(rewards_list)
plt.savefig("./out/rewards.png")
plt.show()


# save
import json
with open('./out/qmatrix.json', 'w') as f:
    json.dump(Q_matrix.tolist(), f)

