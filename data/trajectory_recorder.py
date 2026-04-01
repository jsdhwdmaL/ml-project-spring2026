"""Trajectory recorder for human data collection.
Records raw trajectory data during episode execution and builds action chunks.
"""

from typing import Dict, Optional

import numpy as np


class TrajectoryRecorder:
    """Records raw trajectory data during episode and builds action chunks on finalize.

    Stores raw per-step data and builds chunked actions in finalize() using the
    StreamingChunkBuffer pattern from replay_buffer.py.
    """

    def __init__(
        self,
        horizon: int = 16,
        n_obs_steps: int = 2,
        state_dim: int = 18,
        act_dim: int = 2,
    ):
        """Initialize the trajectory recorder.

        Args:
            horizon: Action chunk horizon (default 16)
            n_obs_steps: Number of observation history steps (default 2)
            state_dim: State dimension (default 18 for PushT)
            act_dim: Action dimension (default 2 for PushT)
        """
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        """Clear recorded data."""
        self.observations = []  # (n_obs_steps, state_dim) frame-stacked states
        self.raw_actions = []  # (act_dim,) single raw action per step
        self.rewards = []  # float, actual env reward
        self.dones = []  # bool, termination flag
        self.success = []  # bool, success flag
        self.is_human = []  # bool
        self.is_pure_teleop = []  # bool
        self.images = []  # Optional (96, 96, 3) uint8

    def record_step(
        self,
        obs_state: np.ndarray,
        raw_action: np.ndarray,
        reward: float,
        done: bool,
        success: bool,
        is_human: bool,
        is_pure_teleop: bool = False,
        image: Optional[np.ndarray] = None,
    ):
        """Record one step of raw data.

        Args:
            obs_state: Observation state (n_obs_steps, state_dim)
            raw_action: Single raw action (act_dim,) - NOT chunked
            reward: Reward from environment
            done: Whether episode terminated/truncated
            success: Whether this transition is successful
            is_human: Whether this step was human-controlled
            is_pure_teleop: Whether this step is from pure teleoperation mode
            image: Optional image observation (H, W, 3)
        """
        self.observations.append(obs_state.copy())
        self.raw_actions.append(raw_action.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.success.append(success)
        self.is_human.append(is_human)
        self.is_pure_teleop.append(is_pure_teleop)
        if image is not None:
            self.images.append(image.copy())

    def _build_action_chunks(self) -> tuple:
        """Build action chunks from raw actions.

        Uses StreamingChunkBuffer pattern: for each timestep t, the action chunk
        contains actions from t to t+horizon-1, with padding for steps beyond
        episode end.

        Returns:
            Tuple of (action_chunks, action_is_pad) arrays
        """
        T = len(self.raw_actions)
        action_chunks = np.zeros((T, self.horizon, self.act_dim), dtype=np.float64)
        action_is_pad = np.zeros((T, self.horizon), dtype=bool)

        for t in range(T):
            for j in range(self.horizon):
                future_t = t + j
                if future_t < T:
                    action_chunks[t, j] = self.raw_actions[future_t]
                else:
                    # Pad with last valid action
                    action_chunks[t, j] = self.raw_actions[-1]
                    action_is_pad[t, j] = True

        return action_chunks, action_is_pad

    def finalize(
        self,
        env_seed: int,
        trial_idx: int,
        policy_seed: int,
        terminated: bool,
        truncated: bool,
        success: bool,
        is_pure_teleop_episode: bool = False,
    ) -> Dict:
        """Finalize and return trajectory data compatible with dataloader.

        Builds action chunks from raw actions and formats all data for NPZ storage.

        Args:
            env_seed: Environment seed
            trial_idx: Trial index within this seed
            policy_seed: Policy random seed
            terminated: Whether episode terminated (goal reached)
            truncated: Whether episode was truncated (time limit)
            success: Whether episode was successful
            is_pure_teleop_episode: Whether this episode is pure teleoperation

        Returns:
            Dict with trajectory data in dataloader-compatible format
        """
        T = len(self.observations)

        # Build action chunks
        action_chunks, action_is_pad = self._build_action_chunks()

        # Build frame indices
        frame_index = np.arange(T, dtype=np.int64)

        # Build done array (only last step is done)
        # Build success array (success only at final step if successful)
        done_array = np.array(self.dones, dtype=bool)
        success_array = np.array(self.success, dtype=bool)

        if len(done_array) != T:
            done_array = np.zeros(T, dtype=bool)
            if T > 0:
                done_array[-1] = terminated or truncated

        if len(success_array) != T:
            success_array = np.zeros(T, dtype=bool)
            if T > 0 and success:
                success_array[-1] = True

        is_pure_teleop_array = np.array(self.is_pure_teleop, dtype=bool)
        if len(is_pure_teleop_array) != T:
            is_pure_teleop_array = np.full(T, is_pure_teleop_episode, dtype=bool)

        data = {
            # Core fields (required for dataloader merge)
            'observation.state': np.array(self.observations, dtype=np.float64),
            'action': action_chunks,  # (T, horizon, act_dim) - CHUNKED
            'action_is_pad': action_is_pad,  # (T, horizon)
            'frame_index': frame_index,  # (T,)
            'environment.raw_reward': np.array(self.rewards, dtype=np.float64),  # Raw env rewards
            'next.done': done_array,
            'next.success': success_array,

            # Episode metadata (scalar)
            'env_seed': np.array(env_seed, dtype=np.int64),
            'trial_idx': np.array(trial_idx, dtype=np.int64),
            'policy_seed': np.array(policy_seed, dtype=np.int64),
            'terminated': np.array(terminated, dtype=bool),
            'truncated': np.array(truncated, dtype=bool),
            'success': np.array(success, dtype=bool),

            # Intervention-specific
            'is_human_intervention': np.array(self.is_human, dtype=bool),
            'is_pure_teleop': is_pure_teleop_array,
        }

        return data

    def get_images(self) -> Optional[np.ndarray]:
        """Get images as array if recorded.

        Returns:
            Array of shape (T, H, W, 3) uint8, or None if no images recorded
        """
        if self.images:
            return np.array(self.images, dtype=np.uint8)
        return None

    def __len__(self) -> int:
        """Return number of recorded steps."""
        return len(self.observations)
