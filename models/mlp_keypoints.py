"""State-only two-hidden-layer MLP for BC on LeRobot ``pusht_keypoints``."""

import torch
import torch.nn as nn

AGENT_POS_DIM = 2
ENV_STATE_DIM = 16
DEFAULT_STATE_DIM = AGENT_POS_DIM + ENV_STATE_DIM
DEFAULT_ACTION_DIM = 2


class BCKeypointsMLP(nn.Module):
    """MLP: state -> hidden -> ReLU -> hidden -> ReLU -> action.

    Shapes follow normalized tensors in ``[-1, 1]`` (min–max) at the interface.
    """

    def __init__(
        self,
        state_dim: int = DEFAULT_STATE_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
