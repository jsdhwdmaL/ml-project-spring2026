"""Trajectory recorder for human data collection.

Records raw per-step trajectory data and writes LeRobot-style per-step fields.
"""

from typing import Dict, Optional

import numpy as np


class TrajectoryRecorder:
    """Records raw per-step trajectory data and formats it at finalize()."""

    def __init__(
        self,
        state_dim: int = 2,
        act_dim: int = 2,
    ):
        """Initialize the trajectory recorder.

        Args:
            state_dim: State dimension (default 2 for LeRobot-style PushT)
            act_dim: Action dimension (default 2 for PushT)
        """
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        """Clear recorded data."""
        self.observations = []  # (state_dim,) states
        self.raw_actions = []  # (act_dim,) single raw action per step
        self.rewards = []  # float
        self.dones = []  # bool, termination flag
        self.success = []  # bool, success flag
        self.is_human = []  # bool
        self.images = []  # Optional (96, 96, 3) uint8

    def record_step(
        self,
        obs_state: np.ndarray,
        raw_action: np.ndarray,
        reward: float,
        done: bool,
        success: bool,
        is_human: bool,
        image: Optional[np.ndarray] = None,
    ):
        """Record one step of raw data.

        Args:
            obs_state: Observation state (state_dim,)
            raw_action: Single raw action (act_dim,)
            reward: Reward from environment
            done: Whether episode terminated/truncated
            success: Whether this transition is successful
            is_human: Whether this step was human-controlled
            image: Optional image observation (H, W, 3)
        """
        self.observations.append(obs_state.copy())
        self.raw_actions.append(raw_action.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.success.append(success)
        self.is_human.append(is_human)
        if image is not None:
            self.images.append(image.copy())

    def finalize(
        self,
        env_seed: int,
        trial_idx: int,
        policy_seed: int,
        terminated: bool,
        truncated: bool,
        success: bool,
    ) -> Dict:
        """Finalize and return trajectory data in per-step raw schema.

        Args:
            env_seed: Environment seed
            trial_idx: Trial index within this seed
            policy_seed: Policy random seed
            terminated: Whether episode terminated (goal reached)
            truncated: Whether episode was truncated (time limit)
            success: Whether episode was successful

        Returns:
            Dict with trajectory data in per-step raw format
        """
        T = len(self.observations)

        # Build frame indices
        frame_index = np.arange(T, dtype=np.int64)
        timestamp = frame_index.astype(np.float32)

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

        data = {
            # Core per-step fields
            'observation.state': np.array(self.observations, dtype=np.float32),
            'action': np.array(self.raw_actions, dtype=np.float32),
            'frame_index': frame_index,  # (T,)
            'timestamp': timestamp,
            'next.reward': np.array(self.rewards, dtype=np.float32),
            'next.done': done_array,
            'next.success': success_array,
            'episode_index': np.zeros(T, dtype=np.int64),
            'index': frame_index.copy(),
            'task_index': np.zeros(T, dtype=np.int64),
            'is_human_intervention': np.array(self.is_human, dtype=bool),

            # Episode metadata (scalar)
            'env_seed': np.array(env_seed, dtype=np.int64),
            'trial_idx': np.array(trial_idx, dtype=np.int64),
            'policy_seed': np.array(policy_seed, dtype=np.int64),
            'terminated': np.array(terminated, dtype=bool),
            'truncated': np.array(truncated, dtype=bool),
            'success': np.array(success, dtype=bool),
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
