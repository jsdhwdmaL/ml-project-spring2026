"""Config-driven ACT policy aligned with LeRobot's ACT implementation.

This module mirrors :class:`models.act.ACTPolicy` but exposes every knob in
LeRobot's ``ACTConfig`` so a checkpoint produced from the LeRobot trainer
can be reproduced architecturally.

Compared to ``models.act.ACTPolicy`` the following extra knobs are
plumbed through:

- ``dim_feedforward`` (was hard-coded to ``hidden_dim * 4``).
- ``feedforward_activation`` (was hard-coded to ``gelu``).
- ``dropout`` (was hard-coded to ``0.1``).
- ``pre_norm`` (was always post-norm; LeRobot defaults to post-norm too).
- ``n_vae_encoder_layers`` (separate from the context encoder depth).
- ``use_vae`` (when ``False``, no CVAE encoder is built; latent is zero).
- ``vision_backbone`` (``resnet18``/``resnet34``/``resnet50``).
- ``pretrained_backbone_weights`` (e.g. ``"ResNet18_Weights.IMAGENET1K_V1"``).
- ``replace_final_stride_with_dilation``.
- ``n_obs_steps`` — number of stacked observation timesteps fed in as
  additional context tokens (vector obs) or repeated spatial maps (image obs).
- Multi-key observation inputs via ``input_shapes`` (each key becomes its own
  projected token; LeRobot uses ``observation.state`` and
  ``observation.environment_state`` for the keypoints task).

Inference-only fields (``n_action_steps``, ``temporal_ensemble_momentum``)
are stored on the module as metadata for the eval/dagger scripts; they do
not change the forward pass itself.

Forward signature::

    pred_actions, mu, logvar = model(observations, action_chunk=None)

where ``observations`` is a dict ``{key: tensor}`` matching the keys in
``input_shapes``. Vector tensors may be ``(B, *shape)`` (when
``n_obs_steps==1``) or ``(B, n_obs_steps, *shape)``. Image tensors may be
``(B, C, H, W)`` or ``(B, n_obs_steps, C, H, W)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

def sinusoidal_position_embedding(
    seq_len: int, dim: int, device: torch.device
) -> torch.Tensor:
    """Returns a (1, seq_len, dim) sinusoidal positional embedding."""
    position = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


# ---------------------------------------------------------------------------
# Vision backbone
# ---------------------------------------------------------------------------

_BACKBONE_REGISTRY = {
    "resnet18": (models.resnet18, 512),
    "resnet34": (models.resnet34, 512),
    "resnet50": (models.resnet50, 2048),
}


def _resolve_backbone_weights(weights_str: Optional[str]):
    """Parse a torchvision Weights enum from a dotted string.

    e.g. ``"ResNet18_Weights.IMAGENET1K_V1"`` →
    ``torchvision.models.ResNet18_Weights.IMAGENET1K_V1``.
    """
    if not weights_str:
        return None
    if "." not in weights_str:
        return weights_str  # let torchvision interpret e.g. "DEFAULT"
    enum_name, member = weights_str.split(".", 1)
    enum_cls = getattr(models, enum_name, None)
    if enum_cls is None:
        return weights_str
    return getattr(enum_cls, member)


def _build_vision_backbone(
    name: str,
    pretrained_weights: Optional[str],
    replace_final_stride_with_dilation: bool,
) -> Tuple[nn.Sequential, int]:
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unsupported vision_backbone={name!r}; "
            f"supported: {sorted(_BACKBONE_REGISTRY)}"
        )
    ctor, out_channels = _BACKBONE_REGISTRY[name]
    weights = _resolve_backbone_weights(pretrained_weights)
    kwargs = {"weights": weights}
    if replace_final_stride_with_dilation:
        kwargs["replace_stride_with_dilation"] = [False, False, True]
    backbone = ctor(**kwargs)
    return nn.Sequential(*list(backbone.children())[:-2]), out_channels


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ACTLeRobotConfig:
    """Mirror of LeRobot's ACTConfig (subset that affects the model).

    All defaults match LeRobot's ACT defaults so that a config dict from
    a LeRobot run can be instantiated unchanged via :meth:`from_dict`.
    """

    # I/O shapes (multi-key observations + single action key).
    input_shapes: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "observation.state": [2],
            "observation.environment_state": [16],
        }
    )
    output_shapes: Dict[str, List[int]] = field(
        default_factory=lambda: {"action": [2]}
    )
    input_normalization_modes: Dict[str, str] = field(
        default_factory=lambda: {
            "observation.state": "min_max",
            "observation.environment_state": "min_max",
        }
    )
    output_normalization_modes: Dict[str, str] = field(
        default_factory=lambda: {"action": "min_max"}
    )

    # Architecture.
    chunk_size: int = 16
    n_action_steps: int = 16
    n_obs_steps: int = 1
    dim_model: int = 512
    dim_feedforward: int = 3200
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_vae_encoder_layers: int = 4
    latent_dim: int = 32
    dropout: float = 0.1
    feedforward_activation: str = "relu"
    pre_norm: bool = False
    use_vae: bool = True

    # Vision.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: Optional[str] = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # Loss / inference (stored as metadata, not used inside forward).
    kl_weight: float = 10.0
    temporal_ensemble_momentum: Optional[float] = None

    @classmethod
    def from_dict(cls, cfg: dict) -> "ACTLeRobotConfig":
        """Build a config from a (possibly partial) LeRobot-style dict.

        Unknown keys are ignored to allow forward-compatibility with
        future LeRobot fields (e.g. ``optimizer_lr``).
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cfg.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_image_key(key: str, shape: List[int]) -> bool:
    """LeRobot convention: image keys start with ``observation.image``.

    Fallback: any 3-D (C, H, W) shape with first dim 1/3/4.
    """
    if key.startswith("observation.image"):
        return True
    if len(shape) == 3 and shape[0] in (1, 3, 4):
        return True
    return False


def _make_encoder_layer(cfg: ACTLeRobotConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=cfg.dim_model,
        nhead=cfg.n_heads,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        activation=cfg.feedforward_activation,
        batch_first=True,
        norm_first=cfg.pre_norm,
    )


def _make_decoder_layer(cfg: ACTLeRobotConfig) -> nn.TransformerDecoderLayer:
    return nn.TransformerDecoderLayer(
        d_model=cfg.dim_model,
        nhead=cfg.n_heads,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        activation=cfg.feedforward_activation,
        batch_first=True,
        norm_first=cfg.pre_norm,
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class ACTLeRobotPolicy(nn.Module):
    """ACT policy whose architecture is fully driven by an :class:`ACTLeRobotConfig`."""

    def __init__(self, config: ACTLeRobotConfig):
        super().__init__()
        self.config = config

        if "action" not in config.output_shapes:
            raise ValueError("output_shapes must contain an 'action' key")
        action_shape = config.output_shapes["action"]
        if len(action_shape) != 1:
            raise ValueError(f"Only 1-D action shapes supported; got {action_shape}")
        self.action_dim = int(action_shape[0])
        self.horizon = int(config.chunk_size)
        self.hidden_dim = int(config.dim_model)
        self.latent_dim = int(config.latent_dim)
        self.n_obs_steps = int(config.n_obs_steps)

        # Partition input keys into image vs vector observations.
        self.image_keys: List[str] = []
        self.vector_keys: List[str] = []
        self.vector_dims: Dict[str, int] = {}
        for key, shape in config.input_shapes.items():
            if _is_image_key(key, list(shape)):
                self.image_keys.append(key)
            else:
                if len(shape) != 1:
                    raise ValueError(
                        f"Vector observation {key} must be 1-D; got shape {shape}"
                    )
                self.vector_keys.append(key)
                self.vector_dims[key] = int(shape[0])

        # ---------- Vision backbone ----------
        if self.image_keys:
            self.vision_backbone, vision_out_channels = _build_vision_backbone(
                config.vision_backbone,
                config.pretrained_backbone_weights,
                config.replace_final_stride_with_dilation,
            )
            self.vision_proj = nn.Linear(vision_out_channels, self.hidden_dim)
        else:
            self.vision_backbone = None
            self.vision_proj = None

        # ---------- Vector input projections (one per key, shared across obs steps) ----------
        self.vector_input_projs = nn.ModuleDict(
            {
                # Replace '.' with '_' so it's a valid ModuleDict key
                key.replace(".", "_"): nn.Linear(self.vector_dims[key], self.hidden_dim)
                for key in self.vector_keys
            }
        )

        # CVAE encoder is fed the action chunk + a single (most-recent) state.
        # The encoder uses the FIRST vector key as the state input to mirror
        # LeRobot's behavior of conditioning the posterior on robot state.
        self._cvae_state_key: Optional[str] = self.vector_keys[0] if self.vector_keys else None
        if config.use_vae:
            if self._cvae_state_key is None:
                raise ValueError(
                    "use_vae=True requires at least one vector observation in input_shapes"
                )
            cvae_state_dim = self.vector_dims[self._cvae_state_key]
            self.enc_state_proj = nn.Linear(cvae_state_dim, self.hidden_dim)
            self.action_proj = nn.Linear(self.action_dim, self.hidden_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            self.cvae_encoder = nn.TransformerEncoder(
                _make_encoder_layer(config),
                num_layers=int(config.n_vae_encoder_layers),
            )
            self.posterior_mu = nn.Linear(self.hidden_dim, self.latent_dim)
            self.posterior_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        else:
            self.enc_state_proj = None
            self.action_proj = None
            self.cls_token = None
            self.cvae_encoder = None
            self.posterior_mu = None
            self.posterior_logvar = None

        # Latent → token projection (used in both VAE and non-VAE modes).
        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)

        # ---------- Context encoder ----------
        self.context_encoder = nn.TransformerEncoder(
            _make_encoder_layer(config),
            num_layers=int(config.n_encoder_layers),
        )

        # ---------- Action decoder ----------
        self.decoder = nn.TransformerDecoder(
            _make_decoder_layer(config),
            num_layers=int(config.n_decoder_layers),
        )
        query = sinusoidal_position_embedding(
            self.horizon, self.hidden_dim, torch.device("cpu")
        ).squeeze(0)  # (horizon, D)
        self.register_buffer("query_embed", query)

        # ---------- Head ----------
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # Init & helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _flatten_obs_steps(self, x: torch.Tensor, expected_trailing: Tuple[int, ...]) -> torch.Tensor:
        """Reshape observations to (B*n_obs_steps, *trailing) and verify shape.

        Accepts ``x`` shaped either ``(B, *trailing)`` (when n_obs_steps==1)
        or ``(B, n_obs_steps, *trailing)``.
        """
        if x.shape[1:] == expected_trailing:
            if self.n_obs_steps != 1:
                raise ValueError(
                    f"Expected obs with n_obs_steps={self.n_obs_steps} but got "
                    f"shape {tuple(x.shape)} (no time axis)."
                )
            return x  # (B, *trailing)
        if x.dim() >= 3 and x.shape[1] == self.n_obs_steps and x.shape[2:] == expected_trailing:
            B, T = x.shape[0], x.shape[1]
            return x.reshape(B * T, *expected_trailing)
        raise ValueError(
            f"Observation shape {tuple(x.shape)} does not match expected "
            f"trailing dims {expected_trailing} with n_obs_steps={self.n_obs_steps}."
        )

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Run backbone, project, add 2-D sinusoidal pos-emb.

        ``image`` is ``(B, C, H, W)`` (already flattened over obs steps).
        Returns ``(B, N, hidden_dim)`` with ``N = H'*W'``.
        """
        if self.vision_backbone is None:
            raise RuntimeError("_encode_image called but no vision backbone is configured")
        feat = self.vision_backbone(image)
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        feat = self.vision_proj(feat)

        half = self.hidden_dim // 2
        pe_h = sinusoidal_position_embedding(H, half, feat.device)
        pe_w = sinusoidal_position_embedding(W, half, feat.device)
        pe_h = pe_h.expand(1, H, -1).unsqueeze(2).expand(1, H, W, -1)
        pe_w = pe_w.expand(1, W, -1).unsqueeze(1).expand(1, H, W, -1)
        pe = torch.cat([pe_h, pe_w], dim=-1).reshape(1, H * W, self.hidden_dim)
        return feat + pe

    def _gather_vector_token(self, observations: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        """Project one vector obs key into ``(B, n_obs_steps, hidden_dim)``."""
        x = observations[key]
        dim = self.vector_dims[key]
        x_flat = self._flatten_obs_steps(x, (dim,))  # (B*T, dim) or (B, dim)
        proj = self.vector_input_projs[key.replace(".", "_")](x_flat)
        if proj.dim() == 2 and proj.shape[0] != x.shape[0]:
            B = x.shape[0]
            return proj.reshape(B, self.n_obs_steps, self.hidden_dim)
        return proj.unsqueeze(1)  # (B, 1, hidden_dim)

    def _gather_image_tokens(
        self, observations: Dict[str, torch.Tensor], key: str
    ) -> torch.Tensor:
        """Project one image obs key into ``(B, n_obs_steps*N, hidden_dim)``."""
        x = observations[key]
        shape = tuple(self.config.input_shapes[key])
        x_flat = self._flatten_obs_steps(x, shape)  # (B*T, C, H, W) or (B, C, H, W)
        feat = self._encode_image(x_flat)  # (B*T, N, D)
        if feat.shape[0] != x.shape[0]:
            B = x.shape[0]
            N = feat.shape[1]
            return feat.reshape(B, self.n_obs_steps * N, self.hidden_dim)
        return feat  # (B, N, D)

    # ------------------------------------------------------------------
    # CVAE encoder
    # ------------------------------------------------------------------

    def _encode_posterior(
        self, observations: Dict[str, torch.Tensor], action_chunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cvae_encoder is None:
            raise RuntimeError("_encode_posterior called but use_vae=False")
        state_key = self._cvae_state_key
        assert state_key is not None
        state = observations[state_key]
        # Use the most recent obs step for the CVAE state token.
        if state.dim() == 3 and state.shape[1] == self.n_obs_steps:
            state = state[:, -1]
        elif state.shape[1:] != (self.vector_dims[state_key],):
            raise ValueError(
                f"Unexpected shape for {state_key}: {tuple(state.shape)}"
            )

        B = state.size(0)
        state_tok = self.enc_state_proj(state).unsqueeze(1)  # (B, 1, D)
        action_tok = self.action_proj(action_chunk)          # (B, H, D)

        pe = sinusoidal_position_embedding(self.horizon, self.hidden_dim, action_tok.device)
        action_tok = action_tok + pe

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, state_tok, action_tok], dim=1)  # (B, H+2, D)
        enc_out = self.cvae_encoder(seq)
        cls_out = enc_out[:, 0, :]

        mu = self.posterior_mu(cls_out)
        logvar = self.posterior_logvar(cls_out)
        return mu, logvar

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        action_chunk: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not isinstance(observations, dict):
            raise TypeError(
                "ACTLeRobotPolicy.forward expects a dict[str, Tensor] "
                "matching config.input_shapes"
            )
        missing = [k for k in self.config.input_shapes if k not in observations]
        if missing:
            raise KeyError(f"Missing observation keys: {missing}")

        # Infer batch size from any input.
        any_key = next(iter(observations))
        B = observations[any_key].shape[0]
        device = observations[any_key].device
        dtype = observations[any_key].dtype

        # --- 1. Posterior (training) / zero latent (inference) ---
        mu = logvar = None
        if action_chunk is not None and self.config.use_vae:
            mu, logvar = self._encode_posterior(observations, action_chunk)
            latent = self._reparameterize(mu, logvar)
        else:
            latent = torch.zeros(B, self.latent_dim, device=device, dtype=dtype)

        # --- 2. Build context tokens ---
        tokens: List[torch.Tensor] = []
        for key in self.image_keys:
            tokens.append(self._gather_image_tokens(observations, key))
        for key in self.vector_keys:
            tokens.append(self._gather_vector_token(observations, key))
        tokens.append(self.latent_proj(latent).unsqueeze(1))  # (B, 1, D)

        context = torch.cat(tokens, dim=1)
        memory = self.context_encoder(context)

        # --- 3. Decode action chunk ---
        query = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, horizon, D)
        decoded = self.decoder(tgt=query, memory=memory)
        pred_action_chunk = self.action_head(decoded)
        return pred_action_chunk, mu, logvar

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=1).mean()

    @staticmethod
    def masked_l1_loss(
        pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor
    ) -> torch.Tensor:
        valid = (~is_pad).unsqueeze(-1)
        abs_error = torch.abs(pred - target)
        valid_error = abs_error * valid.float()
        denom = valid.float().sum().clamp_min(1.0) * pred.size(-1)
        return valid_error.sum() / denom

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "ACTLeRobotPolicy":
        return cls(ACTLeRobotConfig.from_dict(cfg))


__all__ = [
    "ACTLeRobotConfig",
    "ACTLeRobotPolicy",
    "sinusoidal_position_embedding",
]
