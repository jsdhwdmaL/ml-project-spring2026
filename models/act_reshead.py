import math
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def sinusoidal_position_embedding(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """Returns a (1, seq_len, dim) sinusoidal position embedding."""
    position = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, dim)


class ACTResHeadPolicy(nn.Module):
    """ACT variant that injects teacher signals into the CVAE encoder.

    Teacher signals are only used when action_chunk is provided (training).
    At inference, callers can omit teacher_signal and the model defaults to
    a zero teacher token, matching the "missing hint at test time" setup.
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 2,
        horizon: int = 20,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 7,
        teacher_signal_dim: int = 1,
        teacher_dropout_prob: float = 0.25,
        use_vision: bool = True,
    ):
        super().__init__()
        if teacher_signal_dim < 1:
            raise ValueError(f"teacher_signal_dim must be >= 1, got {teacher_signal_dim}")
        if not (0.0 <= teacher_dropout_prob <= 1.0):
            raise ValueError(f"teacher_dropout_prob must be in [0, 1], got {teacher_dropout_prob}")

        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_vision = use_vision
        self.teacher_signal_dim = teacher_signal_dim
        self.teacher_dropout_prob = teacher_dropout_prob

        # Vision backbone (ResNet18, no final pooling/fc); skipped when use_vision=False.
        if use_vision:
            vision_backbone = models.resnet18(weights="DEFAULT")
            self.vision_backbone = nn.Sequential(*list(vision_backbone.children())[:-2])
            self.vision_proj = nn.Linear(512, hidden_dim)
        else:
            self.vision_backbone = None
            self.vision_proj = None

        # State / action projections
        self.enc_state_proj = nn.Linear(state_dim, hidden_dim)  # CVAE encoder only
        self.dec_state_proj = nn.Linear(state_dim, hidden_dim)  # CVAE decoder only
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Teacher-signal projection into a dedicated encoder token.
        self.teacher_proj = nn.Sequential(
            nn.Linear(teacher_signal_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.teacher_token_bias = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # CVAE Encoder
        # Inputs: [CLS] + state + teacher + action_chunk (horizon), horizon+3 tokens total.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.cvae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.posterior_mu = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim, latent_dim)

        # Latent projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # CVAE Decoder context encoder
        dec_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.context_encoder = nn.TransformerEncoder(dec_enc_layer, num_layers=num_encoder_layers)

        # Transformer decoder: fixed sinusoidal queries attend to context.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        query = sinusoidal_position_embedding(horizon, hidden_dim, torch.device("cpu")).squeeze(0)
        self.register_buffer("query_embed", query)

        # Action prediction head
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.teacher_token_bias, std=0.02)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.use_vision or self.vision_backbone is None:
            raise RuntimeError("_encode_image called with use_vision=False")
        feat = self.vision_backbone(image)  # (B, 512, H', W')
        bsz, channels, height, width = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
        feat = self.vision_proj(feat)

        pe_h = sinusoidal_position_embedding(height, self.hidden_dim // 2, feat.device)
        pe_w = sinusoidal_position_embedding(width, self.hidden_dim // 2, feat.device)
        pe_h = pe_h.expand(1, height, -1).unsqueeze(2).expand(1, height, width, -1)
        pe_w = pe_w.expand(1, width, -1).unsqueeze(1).expand(1, height, width, -1)
        pe = torch.cat([pe_h, pe_w], dim=-1).reshape(1, height * width, self.hidden_dim)
        return feat + pe

    def _prepare_teacher_signal(
        self,
        teacher_signal: Optional[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if teacher_signal is None:
            out = torch.zeros(batch_size, self.teacher_signal_dim, dtype=dtype, device=device)
        else:
            if teacher_signal.ndim != 2:
                raise ValueError(
                    f"teacher_signal must be rank-2 (B, teacher_signal_dim), got {tuple(teacher_signal.shape)}"
                )
            if teacher_signal.size(1) != self.teacher_signal_dim:
                raise ValueError(
                    f"teacher_signal dim mismatch: expected {self.teacher_signal_dim}, got {teacher_signal.size(1)}"
                )
            out = teacher_signal.to(device=device, dtype=dtype)

        # Randomly drop teacher signal during training so model remains robust
        # when no hint is available at test time.
        if self.training and self.teacher_dropout_prob > 0.0:
            keep = (torch.rand(batch_size, 1, device=device) > self.teacher_dropout_prob).to(dtype=dtype)
            out = out * keep
        return out

    def _encode_posterior(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        teacher_signal: Optional[torch.Tensor] = None,
    ):
        bsz = state.size(0)
        state_tok = self.enc_state_proj(state).unsqueeze(1)
        action_tok = self.action_proj(action_chunk)

        pe = sinusoidal_position_embedding(self.horizon, self.hidden_dim, action_tok.device)
        action_tok = action_tok + pe

        teacher_vec = self._prepare_teacher_signal(
            teacher_signal=teacher_signal,
            batch_size=bsz,
            dtype=state.dtype,
            device=state.device,
        )
        teacher_tok = self.teacher_proj(teacher_vec).unsqueeze(1) + self.teacher_token_bias.expand(bsz, -1, -1)

        cls = self.cls_token.expand(bsz, -1, -1)
        seq = torch.cat([cls, state_tok, teacher_tok, action_tok], dim=1)

        enc_out = self.cvae_encoder(seq)
        cls_out = enc_out[:, 0, :]
        mu = self.posterior_mu(cls_out)
        logvar = self.posterior_logvar(cls_out)
        return mu, logvar

    def forward(
        self,
        image: Optional[torch.Tensor],
        state: torch.Tensor,
        action_chunk: Optional[torch.Tensor] = None,
        teacher_signal: Optional[torch.Tensor] = None,
    ):
        bsz = state.size(0)
        dev = state.device
        dtype = state.dtype

        mu = logvar = None
        if action_chunk is not None:
            mu, logvar = self._encode_posterior(state, action_chunk, teacher_signal=teacher_signal)
            latent = self._reparameterize(mu, logvar)
        else:
            latent = torch.zeros(bsz, self.latent_dim, dtype=dtype, device=dev)

        state_tok = self.dec_state_proj(state).unsqueeze(1)
        z_tok = self.latent_proj(latent).unsqueeze(1)

        if self.use_vision:
            if image is None:
                raise ValueError("image is required when use_vision=True")
            vision_tokens = self._encode_image(image)
            context = torch.cat([vision_tokens, state_tok, z_tok], dim=1)
        else:
            context = torch.cat([state_tok, z_tok], dim=1)

        memory = self.context_encoder(context)
        query = self.query_embed.unsqueeze(0).expand(bsz, -1, -1)
        decoded = self.decoder(tgt=query, memory=memory)
        pred_action_chunk = self.action_head(decoded)

        return pred_action_chunk, mu, logvar

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=1).mean()
