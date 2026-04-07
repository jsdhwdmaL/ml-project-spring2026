import math
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


class ACTPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 2,
        horizon: int = 20,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Vision backbone (ResNet18, no final pooling/fc)
        vision_backbone = models.resnet18(weights="DEFAULT")
        self.vision_backbone = nn.Sequential(*list(vision_backbone.children())[:-2])
        # ResNet18 produces (B, 512, H', W') feature maps
        self.vision_proj = nn.Linear(512, hidden_dim)

        # State / action projections
        self.enc_state_proj = nn.Linear(state_dim, hidden_dim)  # CVAE encoder only
        self.dec_state_proj = nn.Linear(state_dim, hidden_dim)  # CVAE decoder only
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # CVAE Encoder (training only)
        # Inputs: [CLS] (1) + state (1) + action_chunk (horizon), horizon+2 tokens total
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.cvae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.posterior_mu     = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim, latent_dim)

        # Latent projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # CVAE Decoder
        # Transformer encoder fuses: vision tokens + state token + z token
        dec_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(dec_enc_layer, num_layers=num_encoder_layers)

        # Transformer decoder: fixed sinusoidal queries attend to context
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Fixed sinusoidal query embeddings for the transformer decoder
        query = sinusoidal_position_embedding(horizon, hidden_dim, torch.device("cpu")).squeeze(0)  # (horizon, D)
        self.register_buffer("query_embed", query)

        # Action prediction head
        self.action_head = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    # Helpers

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Run ResNet backbone, project, and add 2-D sinusoidal pos-emb.

        Returns a sequence of shape (B, N, hidden_dim) where N = H' * W'.
        """
        feat = self.vision_backbone(image)          # (B, 512, H', W')
        B, C, H, W = feat.shape
        # flatten spatial dims to (B, H'*W', 512)
        feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        feat = self.vision_proj(feat)               # (B, N, hidden_dim)

        # 2-D sinusoidal positional embedding (treat H' and W' independently,
        # then concatenate and project to hidden_dim)
        pe_h = sinusoidal_position_embedding(H, self.hidden_dim // 2, feat.device)  # (1, H, D/2)
        pe_w = sinusoidal_position_embedding(W, self.hidden_dim // 2, feat.device)  # (1, W, D/2)
        pe_h = pe_h.expand(1, H, -1).unsqueeze(2).expand(1, H, W, -1)  # (1,H,W,D/2)
        pe_w = pe_w.expand(1, W, -1).unsqueeze(1).expand(1, H, W, -1)  # (1,H,W,D/2)
        pe = torch.cat([pe_h, pe_w], dim=-1).reshape(1, H * W, self.hidden_dim)  # (1,N,D)
        feat = feat + pe
        return feat  # (B, N, hidden_dim)

    # CVAE Encoder

    def _encode_posterior(self, state: torch.Tensor, action_chunk: torch.Tensor):
        """Encode (state, action_chunk) → (mu, logvar) via TransformerEncoder.

        Inputs:
            state:        (B, state_dim)
            action_chunk: (B, horizon, action_dim)
        """
        B = state.size(0)

        # Project state → (B, 1, hidden_dim)
        state_tok = self.enc_state_proj(state).unsqueeze(1)

        # Project action chunk → (B, horizon, hidden_dim)
        action_tok = self.action_proj(action_chunk)

        # Add sinusoidal positional embedding to action tokens only
        pe = sinusoidal_position_embedding(self.horizon, self.hidden_dim, action_tok.device)
        action_tok = action_tok + pe  # (B, horizon, D)

        # Prepend CLS token and state token → (B, horizon+2, hidden_dim)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, state_tok, action_tok], dim=1)  # (B, H+2, D)

        # Encode; take CLS output (index 0) to predict z distribution
        enc_out = self.cvae_encoder(seq)       # (B, H+2, D)
        cls_out = enc_out[:, 0, :]             # (B, D)

        mu     = self.posterior_mu(cls_out)     # (B, latent_dim)
        logvar = self.posterior_logvar(cls_out) # (B, latent_dim)
        return mu, logvar

    # Forward

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        action_chunk: torch.Tensor | None = None,
    ):
        B = image.size(0)

        # 1. CVAE Encoder (training) / zero latent (test)
        mu = logvar = None
        if action_chunk is not None:
            mu, logvar = self._encode_posterior(state, action_chunk)
            latent = self._reparameterize(mu, logvar)
        else:
            latent = torch.zeros(B, self.latent_dim, dtype=image.dtype, device=image.device)

        # 2. Build context for CVAE Decoder

        # Vision tokens with 2-D positional embedding: (B, N, D)
        vision_tokens = self._encode_image(image)

        # State token: (B, 1, D)
        state_tok = self.dec_state_proj(state).unsqueeze(1)

        # Latent (z) token: (B, 1, D)
        z_tok = self.latent_proj(latent).unsqueeze(1)

        # Concatenate all context tokens (vision already has pos emb from _encode_image)
        context = torch.cat([vision_tokens, state_tok, z_tok], dim=1)  # (B, N+2, D)

        # Fuse all context tokens with a TransformerEncoder
        memory = self.context_encoder(context)  # (B, N+2, D)

        # 3. CVAE Decoder with fixed sinusoidal query embeddings
        # Queries: fixed sinusoidal embedding expanded over batch (B, horizon, D)
        query = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        decoded = self.decoder(tgt=query, memory=memory)  # (B, horizon, D)

        # 4. Project to action space
        pred_action_chunk = self.action_head(decoded)  # (B, horizon, action_dim)

        return pred_action_chunk, mu, logvar

    # Loss helpers

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=1).mean()
