import torch
import torch.nn as nn
from torchvision import models


class ACTPolicy(nn.Module):
	def __init__(
		self,
		state_dim: int = 2,
		action_dim: int = 2,
		horizon: int = 20,
		hidden_dim: int = 256,
		latent_dim: int = 32,
		nhead: int = 8,
		num_decoder_layers: int = 6,
	):
		super().__init__()
		self.horizon = horizon
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim

		vision_backbone = models.resnet18(weights="DEFAULT")
		self.vision_backbone = nn.Sequential(*list(vision_backbone.children())[:-1])

		self.state_encoder = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
		)

		self.vision_proj = nn.Linear(512, hidden_dim)

		self.posterior_net = nn.Sequential(
			nn.Linear(hidden_dim + horizon * action_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
		)
		self.posterior_mu = nn.Linear(hidden_dim, latent_dim)
		self.posterior_logvar = nn.Linear(hidden_dim, latent_dim)

		self.latent_proj = nn.Linear(latent_dim, hidden_dim)
		self.condition_fuse = nn.Sequential(
			nn.Linear(hidden_dim * 3, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
		)

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
		self.query_embed = nn.Parameter(torch.randn(1, horizon, hidden_dim) * 0.02)
		self.action_head = nn.Linear(hidden_dim, action_dim)

	def _encode_obs(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
		image_features = self.vision_backbone(image).view(image.size(0), -1)
		image_features = self.vision_proj(image_features)
		state_features = self.state_encoder(state)
		return image_features, state_features

	def _encode_posterior(self, obs_embed: torch.Tensor, action_chunk: torch.Tensor):
		action_flat = action_chunk.reshape(action_chunk.size(0), -1)
		posterior_input = torch.cat([obs_embed, action_flat], dim=1)
		hidden = self.posterior_net(posterior_input)
		mu = self.posterior_mu(hidden)
		logvar = self.posterior_logvar(hidden)
		return mu, logvar

	@staticmethod
	def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def forward(
		self,
		image: torch.Tensor,
		state: torch.Tensor,
		action_chunk: torch.Tensor | None = None,
		sample_posterior: bool = True,
	):
		image_features, state_features = self._encode_obs(image, state)
		obs_embed = image_features + state_features

		mu = None
		logvar = None
		if action_chunk is not None:
			mu, logvar = self._encode_posterior(obs_embed, action_chunk)
			latent = self._reparameterize(mu, logvar) if sample_posterior else mu
		else:
			latent = torch.zeros(
				(image.size(0), self.latent_dim),
				dtype=image.dtype,
				device=image.device,
			)

		latent_features = self.latent_proj(latent)
		memory_token = self.condition_fuse(torch.cat([image_features, state_features, latent_features], dim=1)).unsqueeze(1)

		query_tokens = self.query_embed.expand(image.size(0), -1, -1)
		decoded = self.decoder(tgt=query_tokens, memory=memory_token)
		pred_action_chunk = self.action_head(decoded)
		return pred_action_chunk, mu, logvar

	@staticmethod
	def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
		return kl.sum(dim=1).mean()
