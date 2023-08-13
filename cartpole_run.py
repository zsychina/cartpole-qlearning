import gymnasium as gym
import json
import numpy as np

with open('./out/qmatrix.json') as f:
    Q_matrix = json.load(f)
Q_matrix = np.array(Q_matrix)

env = gym.make('CartPole-v1', render_mode='human')


upper_bounds = env.observation_space.high
lower_bounds = env.observation_space.low

upper_bounds[1]=3
upper_bounds[3]=10
lower_bounds[1]=-3
lower_bounds[3]=-10

bin_num = [30, 30, 30, 30]

cartPosition_bin = np.linspace(lower_bounds[0], upper_bounds[0], bin_num[0])
cartVelocity_bin = np.linspace(lower_bounds[1], upper_bounds[1], bin_num[1])
poleAngle_bin = np.linspace(lower_bounds[2], upper_bounds[2], bin_num[2])
poleAngleVelocity_bin = np.linspace(lower_bounds[3], upper_bounds[3], bin_num[3])

def get_state_index(state):
    cartPosition_bin_idx = np.digitize(state[0], cartPosition_bin) - 1
    cartVelocity_bin_idx = np.digitize(state[1], cartVelocity_bin) - 1
    poleAngle_bin_idx = np.digitize(state[2], poleAngle_bin) - 1
    poleAngleVelocity_bin_idx = np.digitize(state[3], poleAngleVelocity_bin) - 1
 
    return tuple([cartPosition_bin_idx, cartVelocity_bin_idx, poleAngle_bin_idx, poleAngleVelocity_bin_idx])


def select_action(state):
    state_idx = get_state_index(state)
    action = np.argmax(Q_matrix[state_idx])
        
    return action

observation, info = env.reset()
for i in range(1000):
    action = select_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        
env.close()

