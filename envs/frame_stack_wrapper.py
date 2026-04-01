import gymnasium as gym
from collections import deque
import numpy as np

def space_stack(space: gym.Space, repeat: int):
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise TypeError()

def stack_deque_data(deque_data):
    '''
    WARNING: A huge assumption is that the observation space is either pure np.ndarray,
    or a dict of np.ndarrays that has at most one level of nesting.
    i.e. obs["state"]["left/gripper_pos"] or obs["images"]["left/top"]
    '''
    stacked_data = {k: {} for k in deque_data[0].keys()}
    for key in stacked_data.keys():
        if isinstance(deque_data[0][key], np.ndarray):
            stacked_data[key] = np.stack([item[key] for item in deque_data], axis=0)
        elif isinstance(deque_data[0][key], dict):
            stacked_data[key] = {}
            for sub_key in deque_data[0][key].keys():
                stacked_data[key][sub_key] = np.stack(
                    [item[key][sub_key] for item in deque_data], axis=0
                )
        else:
            raise ValueError(f"Unsupported data type for key {key}: {type(deque_data[0][key])}")
    return stacked_data

class FrameStackWrapperEnv(gym.ObservationWrapper):
    """
    A wrapper that returns a stacked observation of exactly two frames 
    that are 9 steps apart: (t-9, t). 
    - For the first 9 steps after reset, we replicate the initial frame 
      to fill the older slot.
    """
    def __init__(self, env, n_frames=2, gap=9):
        super(FrameStackWrapperEnv, self).__init__(env)
        self.gap = gap
        self.n_frames = n_frames  # We want exactly 2 time steps: t-gap and t
        # We'll store up to gap+1 frames so we can always index t-gap
        self.frames = deque(maxlen=self.gap + 1)
        # The observation space will be "stacked" along a new leading dimension of size 2
        self.observation_space = space_stack(self.env.observation_space, self.n_frames)
        # self.stack_timestamp = 0

    def reset(self, **kwargs):
        """Reset the environment and fill the buffer with the initial observation."""
        self.frames.clear()
        # self.stack_timestamp = 0
        obs, info = self.env.reset(**kwargs)
        # obs['stack_timestamp'] = self.stack_timestamp
        # Fill the buffer with the same obs so that for the first 9 steps, 
        # the 'old' frame is the reset obs.
        for _ in range(self.gap + 1):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        """Take one environment step, add the new frame, then return a stacked observation."""
        obs, reward, done, truncated, info = self.env.step(action)
        # self.stack_timestamp += 1
        # obs['stack_timestamp'] = self.stack_timestamp
        self.frames.append(obs)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        """
        Return two frames: the old one from (current_step - gap) 
        and the newest one from (current_step).
        - frames[-1] is the newest
        - frames[-(gap+1)] is 9 steps older, or the earliest if we haven't stepped enough yet
        """
        if self.n_frames == 1:
            stacked = stack_deque_data([self.frames[-1]])
            return stacked

        # Oldest slot we want (9 steps ago) might be frames[0] if not enough steps passed
        old_obs = self.frames[0] if len(self.frames) < (self.gap + 1) else self.frames[-(self.gap + 1)]
        new_obs = self.frames[-1]

        obs_list = [old_obs, new_obs]  # two time steps: (t-gap, t)
        stacked = stack_deque_data(obs_list)  # shape: (2, ...) for each obs dimension
        return stacked

if __name__ == "__main__":
    import gym_pusht
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos")
    env = FrameStackWrapperEnv(env, n_frames=2, gap=9)

    obs, env_info = env.reset()
    assert obs["pixels"].shape == (2, 96, 96, 3)
    assert obs["agent_pos"].shape == (2, 2)
    action = env.action_space.sample()
    obs, reward, done, truncated, env_info = env.step(action)
    assert obs["pixels"].shape == (2, 96, 96, 3)
    assert obs["agent_pos"].shape == (2, 2)
    print("FrameStackWrapperEnv passed")
    env.close()