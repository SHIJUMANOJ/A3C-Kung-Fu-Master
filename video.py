import imageio
import torch

from environment import make_env
from agent import Agent


def generate_video():

    env = make_env()

    number_actions = env.action_space.n

    agent = Agent(number_actions)

    agent.network.load_state_dict(
        torch.load("checkpoints/a3c_kungfu.pth")
    )

    state, _ = env.reset()

    done = False

    frames = []

    while not done:

        frame = env.render()

        frames.append(frame)

        action = agent.act(state)

        state, reward, terminated, truncated, _ = env.step(action[0])
        done = terminated or truncated

    imageio.mimsave(
        "videos/gameplay.mp4",
        frames,
        fps=30
    )

    print("Video saved!")


if __name__ == "__main__":
    generate_video()