import numpy as np


def evaluate(agent, env, n_episodes=5):

    rewards = []

    for _ in range(n_episodes):

        state, _ = env.reset()

        total_reward = 0

        done = False

        while not done:

            action = agent.act(state)

            state, reward, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated

            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards)