import numpy as np
import tqdm
import torch

from environment import EnvBatch, make_env
from agent import Agent
from evaluate import evaluate
from config import NUMBER_ENVIRONMENTS, TRAINING_STEPS


def train():

    env = make_env()

    number_actions = env.action_space.n

    agent = Agent(number_actions)

    env_batch = EnvBatch(NUMBER_ENVIRONMENTS)

    batch_states = env_batch.reset()

    with tqdm.trange(TRAINING_STEPS) as progress_bar:

        for i in progress_bar:

            batch_actions = agent.act(batch_states)

            (
                batch_next_states,
                batch_rewards,
                batch_dones,
                _
            ) = env_batch.step(batch_actions)

            batch_rewards *= 0.01

            agent.step(
                batch_states,
                batch_actions,
                batch_rewards,
                batch_next_states,
                batch_dones
            )

            batch_states = batch_next_states

            if i % 1000 == 0:

                avg_reward = evaluate(agent, env)

                print(f"Step {i} | Avg Reward: {avg_reward}")

    torch.save(
        agent.network.state_dict(),
        "checkpoints/a3c_kungfu.pth"
    )

    print("Model saved!")


if __name__ == "__main__":
    train()