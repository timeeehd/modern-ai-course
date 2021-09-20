import random

import gym
import numpy as np

# env = gym.make("FrozenLake-v1")
env = gym.make("FrozenLake8x8-v1")
# env.render()

action_size = env.action_space.n

state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))

number_episodes = 250000
max_steps = 400
learning_rate = 0.8
gamma = 0.9

# Exploration parameters
epsilon = 1  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.001  # Minimum exploration probability
decay_rate = 0.00005  # Exponential decay rate for exploration prob

rewards = []
# qtable_history = []
# score_history = []

for episode in range(number_episodes):
    state = env.reset()
    step = 0
    done = False
    total_reward = 0

    for step in range(max_steps):
        trade_off = random.uniform(0, 1)
        if trade_off > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        qtable[state, action] = qtable[state, action] + learning_rate * (
                reward + gamma * np.max(qtable[new_state, :]) -
                qtable[state, action])
        total_reward += reward
        state = new_state
        if done:
            break

    rewards.append(total_reward)
    episode_count = episode + 1

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    # episode_count = episode + 1
    # if episode_count % 10000 == 0:
    #     qtable_history.append(qtable)
    #     score_history.append(sum(rewards) / episode_count)

print("Score over time: " + str(sum(rewards) / number_episodes))
print(qtable)

env.reset()
test_episodes = 1000
rewards = []

for episode in range(test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            print("Score", total_rewards)
            print("Steps: ", step)
            break
        state = new_state
env.close()
print("Total wins: " + str(sum(rewards)) + " out of " + str(test_episodes) + " episodes")
print("Win percentage: " + str(sum(rewards) / test_episodes))
