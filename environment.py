import cv2
import gymnasium as gym
import numpy as np

from gymnasium.spaces import Box
from gymnasium import ObservationWrapper

from config import HEIGHT, WIDTH, N_FRAMES


class PreprocessAtari(ObservationWrapper):

    def __init__(
        self,
        env,
        height=HEIGHT,
        width=WIDTH,
        crop=lambda img: img,
        dim_order='pytorch',
        color=False,
        n_frames=N_FRAMES
    ):

        super(PreprocessAtari, self).__init__(env)

        self.img_size = (height, width)
        self.crop = crop
        self.dim_order = dim_order
        self.color = color
        self.frame_stack = n_frames

        n_channels = 3 * n_frames if color else n_frames

        obs_shape = {
            'tensorflow': (height, width, n_channels),
            'pytorch': (n_channels, height, width)
        }[dim_order]

        self.observation_space = Box(0.0, 1.0, obs_shape)

        self.frames = np.zeros(obs_shape, dtype=np.float32)

    def reset(self, **kwargs):

        self.frames = np.zeros_like(self.frames)

        obs, info = self.env.reset(**kwargs)

        self.update_buffer(obs)

        return self.frames, info

    def observation(self, img):

        img = self.crop(img)

        img = cv2.resize(img, self.img_size)

        if not self.color:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype('float32') / 255.

        if self.color:
            self.frames = np.roll(self.frames, shift=-3, axis=0)
            self.frames[-3:] = img
        else:
            self.frames = np.roll(self.frames, shift=-1, axis=0)
            self.frames[-1] = img

        return self.frames

    def update_buffer(self, obs):
        self.frames = self.observation(obs)


def make_env():

    env_ids = [
        'KungFuMasterNoFrameskip-v0',
        'KungFuMaster-v0',
        'ALE/KungFuMaster-v5'
    ]

    last_error = None
    for env_id in env_ids:
        try:
            env = gym.make(env_id, render_mode='rgb_array')
            break
        except gym.error.Error as exc:
            last_error = exc
    else:
        available = [spec.id for spec in gym.envs.registry.values() if 'KungFuMaster' in spec.id]
        available_msg = '\n'.join(f'  {env_id}' for env_id in available) if available else '  (none found)'
        raise RuntimeError(
            "Unable to create KungFuMaster environment. "
            "Please install the Atari/Gym retro dependencies and use a supported env ID. "
            "Tried: KungFuMasterNoFrameskip-v0, KungFuMaster-v0, ALE/KungFuMaster-v5.\n"
            "Available KungFuMaster envs:\n" + available_msg
        ) from last_error

    env = PreprocessAtari(env)

    return env


class EnvBatch:

    def __init__(self, n_envs=10):
        self.envs = [make_env() for _ in range(n_envs)]

    def reset(self):

        states = []

        for env in self.envs:
            states.append(env.reset()[0])

        return np.array(states)

    def step(self, actions):

        results = [env.step(a) for env, a in zip(self.envs, actions)]

        next_states = []
        rewards = []
        dones = []
        infos = []

        for res in results:
            if len(res) == 5:
                next_state, reward, terminated, truncated, info = res
                done = terminated or truncated
            else:
                next_state, reward, done, info = res

            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        for i in range(len(self.envs)):
            if dones[i]:
                next_states[i] = self.envs[i].reset()[0]

        return next_states, rewards, dones, infos