#!/usr/bin/env python3
"""Train ACT policy on lerobot/pusht using chunked action supervision."""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from data.build_chunk import build_action_chunks_by_episode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from models.act import ACTPolicy


@dataclass
class TrainConfig:
	dataset_id: str = "lerobot/pusht"
	split: str = "train"
	output_dir: str = "models/act"
	seed: int = 42
	val_ratio: float = 0.1
	epochs: int = 50
	batch_size: int = 64
	learning_rate: float = 1e-4
	weight_decay: float = 1e-5
	horizon: int = 20
	hidden_dim: int = 256
	latent_dim: int = 32
	nhead: int = 8
	num_decoder_layers: int = 6
	kl_beta: float = 0.01
	kl_warmup_epochs: int = 50
	kl_ramp_epochs: int = 50
	image_shift_px: int = 4
	ensemble_decay: float = 0.05


class ACTStepDataset(Dataset):
	def __init__(
		self,
		base_dataset: LeRobotDataset,
		step_indices: np.ndarray,
		action_chunks: np.ndarray,
		action_is_pad: np.ndarray,
	):
		self.base_dataset = base_dataset
		self.step_indices = step_indices.astype(np.int64)
		self.action_chunks = action_chunks
		self.action_is_pad = action_is_pad

	def __len__(self) -> int:
		return int(self.step_indices.shape[0])

	def __getitem__(self, idx: int):
		global_idx = int(self.step_indices[idx])
		sample = self.base_dataset[global_idx]
		return {
			"image": sample["observation.image"].float(),
			"state": sample["observation.state"].float(),
			"action_chunk": torch.from_numpy(self.action_chunks[global_idx]).float(),
			"action_is_pad": torch.from_numpy(self.action_is_pad[global_idx]).bool(),
		}


def random_shift_batch(images: torch.Tensor, max_shift: int) -> torch.Tensor:
	if max_shift <= 0:
		return images

	b, c, h, w = images.shape
	pad = int(max_shift)
	padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
	shift_y = torch.randint(0, 2 * pad + 1, (b,), device=images.device)
	shift_x = torch.randint(0, 2 * pad + 1, (b,), device=images.device)

	out = torch.empty_like(images)
	for i in range(b):
		y0 = int(shift_y[i].item())
		x0 = int(shift_x[i].item())
		out[i] = padded[i, :, y0 : y0 + h, x0 : x0 + w]
	return out


def preprocess_image_batch(images: torch.Tensor, image_normalize, random_shift_px: int = 0) -> torch.Tensor:
	images = images.to(dtype=torch.float32)
	if images.ndim != 4:
		raise ValueError(f"Expected image batch with shape (B,C,H,W) or (B,H,W,C), got {tuple(images.shape)}")

	if images.shape[1] != 3 and images.shape[-1] == 3:
		images = images.permute(0, 3, 1, 2)

	if images.numel() > 0 and torch.max(images) > 1.0:
		images = images / 255.0
	if images.shape[-2:] != (96, 96):
		images = F.interpolate(images, size=(96, 96), mode="bilinear", align_corners=False)
	images = random_shift_batch(images, max_shift=random_shift_px)
	images = image_normalize(images)
	return images


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
	valid = (~is_pad).unsqueeze(-1)
	abs_error = torch.abs(pred - target)
	valid_error = abs_error * valid.float()
	denom = valid.float().sum().clamp_min(1.0) * pred.size(-1)
	return valid_error.sum() / denom


def split_episode_indices(episode_index: np.ndarray, val_ratio: float, seed: int):
	unique_eps = np.unique(episode_index)
	rng = np.random.default_rng(seed)
	rng.shuffle(unique_eps)

	n_val_eps = max(1, int(len(unique_eps) * val_ratio)) if val_ratio > 0 else 0
	val_eps = set(unique_eps[:n_val_eps].tolist())

	val_mask = np.array([ep in val_eps for ep in episode_index], dtype=bool)
	train_idx = np.where(~val_mask)[0]
	val_idx = np.where(val_mask)[0]

	if train_idx.size == 0:
		raise ValueError("Empty ACT train split")
	if val_idx.size == 0:
		val_idx = train_idx.copy()
	return train_idx, val_idx


def get_kl_beta(epoch_index: int, config: TrainConfig) -> float:
	if epoch_index < config.kl_warmup_epochs:
		return 0.0
	if config.kl_ramp_epochs <= 0:
		return float(config.kl_beta)

	ramp_step = epoch_index - config.kl_warmup_epochs + 1
	if ramp_step <= config.kl_ramp_epochs:
		progress = ramp_step / float(config.kl_ramp_epochs)
		return float(config.kl_beta) * progress
	return float(config.kl_beta)


def train(config: TrainConfig) -> None:
	os.makedirs(config.output_dir, exist_ok=True)
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
	image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	print(f"Loading dataset {config.dataset_id}...")
	dataset = LeRobotDataset(config.dataset_id)
	if hasattr(dataset, "hf_dataset") and hasattr(dataset.hf_dataset, "keys") and config.split in dataset.hf_dataset.keys():
		dataset.hf_dataset = dataset.hf_dataset[config.split]
	elif hasattr(dataset, "keys") and config.split in dataset.keys():
		dataset = dataset[config.split]

	episode_index = np.array(dataset.hf_dataset["episode_index"]).flatten()
	actions = np.array(dataset.hf_dataset["action"], dtype=np.float32)
	states = np.array(dataset.hf_dataset["observation.state"], dtype=np.float32)

	action_chunks, action_is_pad = build_action_chunks_by_episode(actions, episode_index, config.horizon)
	train_idx, val_idx = split_episode_indices(episode_index, config.val_ratio, config.seed)

	state_mean = states[train_idx].mean(axis=0).astype(np.float32)
	state_std = (states[train_idx].std(axis=0) + 1e-6).astype(np.float32)
	action_mean = actions[train_idx].mean(axis=0).astype(np.float32)
	action_std = (actions[train_idx].std(axis=0) + 1e-6).astype(np.float32)

	train_dataset = ACTStepDataset(dataset, train_idx, action_chunks, action_is_pad)
	val_dataset = ACTStepDataset(dataset, val_idx, action_chunks, action_is_pad)

	train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

	model = ACTPolicy(
		state_dim=2,
		action_dim=2,
		horizon=config.horizon,
		hidden_dim=config.hidden_dim,
		latent_dim=config.latent_dim,
		nhead=config.nhead,
		num_decoder_layers=config.num_decoder_layers,
	).to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

	state_mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
	state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)
	action_mean_t = torch.tensor(action_mean, dtype=torch.float32, device=device)
	action_std_t = torch.tensor(action_std, dtype=torch.float32, device=device)

	best_val = float("inf")

	for epoch in range(1, config.epochs + 1):
		epoch_index = epoch - 1
		epoch_kl_beta = get_kl_beta(epoch_index, config)
		model.train()
		train_loss_sum = 0.0
		train_recon_sum = 0.0
		train_kl_sum = 0.0
		train_batches = 0

		for batch in tqdm(train_loader, desc=f"Train {epoch}/{config.epochs}", leave=False):
			images = preprocess_image_batch(
				batch["image"].to(device),
				image_normalize,
				random_shift_px=config.image_shift_px,
			)
			states_b = batch["state"].to(device)
			action_chunk = batch["action_chunk"].to(device)
			action_is_pad_b = batch["action_is_pad"].to(device)

			states_norm = (states_b - state_mean_t) / state_std_t
			target_actions = (action_chunk - action_mean_t.view(1, 1, -1)) / action_std_t.view(1, 1, -1)

			pred_actions, mu, logvar = model(images, states_norm, target_actions, sample_posterior=True)
			recon_loss = masked_l1_loss(pred_actions, target_actions, action_is_pad_b)
			kl_loss = ACTPolicy.kl_divergence(mu, logvar)
			loss = recon_loss + epoch_kl_beta * kl_loss

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			train_loss_sum += float(loss.item())
			train_recon_sum += float(recon_loss.item())
			train_kl_sum += float(kl_loss.item())
			train_batches += 1

		train_loss = train_loss_sum / max(1, train_batches)
		train_recon = train_recon_sum / max(1, train_batches)
		train_kl = train_kl_sum / max(1, train_batches)

		model.eval()
		val_loss_sum = 0.0
		val_recon_sum = 0.0
		val_kl_sum = 0.0
		val_batches = 0
		with torch.no_grad():
			for batch in tqdm(val_loader, desc=f"Val {epoch}/{config.epochs}", leave=False):
				images = preprocess_image_batch(
					batch["image"].to(device),
					image_normalize,
					random_shift_px=0,
				)
				states_b = batch["state"].to(device)
				action_chunk = batch["action_chunk"].to(device)
				action_is_pad_b = batch["action_is_pad"].to(device)

				states_norm = (states_b - state_mean_t) / state_std_t
				target_actions = (action_chunk - action_mean_t.view(1, 1, -1)) / action_std_t.view(1, 1, -1)

				pred_actions, mu, logvar = model(images, states_norm, target_actions, sample_posterior=False)
				recon_loss = masked_l1_loss(pred_actions, target_actions, action_is_pad_b)
				kl_loss = ACTPolicy.kl_divergence(mu, logvar)
				loss = recon_loss + epoch_kl_beta * kl_loss

				val_loss_sum += float(loss.item())
				val_recon_sum += float(recon_loss.item())
				val_kl_sum += float(kl_loss.item())
				val_batches += 1

		val_loss = val_loss_sum / max(1, val_batches)
		val_recon = val_recon_sum / max(1, val_batches)
		val_kl = val_kl_sum / max(1, val_batches)

		checkpoint = {
			"model_state_dict": model.state_dict(),
			"state_mean": state_mean.astype(np.float32),
			"state_std": state_std.astype(np.float32),
			"action_mean": action_mean.astype(np.float32),
			"action_std": action_std.astype(np.float32),
			"config": asdict(config),
			"epoch": epoch,
			"train_loss": train_loss,
			"train_recon": train_recon,
			"train_kl": train_kl,
			"val_loss": val_loss,
			"val_recon": val_recon,
			"val_kl": val_kl,
		}

		latest_path = os.path.join(config.output_dir, "latest.pt")
		torch.save(checkpoint, latest_path)
		if val_loss < best_val:
			best_val = val_loss
			best_path = os.path.join(config.output_dir, "best.pt")
			torch.save(checkpoint, best_path)

		print(
			f"Epoch {epoch:03d}/{config.epochs} | "
			f"kl_beta={epoch_kl_beta:.6f} | "
			f"train={train_loss:.6f} (recon={train_recon:.6f}, kl={train_kl:.6f}) | "
			f"val={val_loss:.6f} (recon={val_recon:.6f}, kl={val_kl:.6f})"
		)

	with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as file:
		json.dump(asdict(config), file, indent=2)

	np.savez(
		os.path.join(config.output_dir, "normalization_stats.npz"),
		state_mean=state_mean.astype(np.float32),
		state_std=state_std.astype(np.float32),
		action_mean=action_mean.astype(np.float32),
		action_std=action_std.astype(np.float32),
	)

	print(f"Saved ACT artifacts to: {config.output_dir}")
	print(f"Best validation loss: {best_val:.6f}")


def parse_args() -> TrainConfig:
	parser = argparse.ArgumentParser(description="Train ACT on lerobot/pusht")
	parser.add_argument("--dataset_id", type=str, default="lerobot/pusht")
	parser.add_argument("--split", type=str, default="train")
	parser.add_argument("--output_dir", type=str, default="models/act")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--val_ratio", type=float, default=0.1)
	parser.add_argument("--epochs", type=int, default=200)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--learning_rate", type=float, default=1e-5)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--horizon", type=int, default=50)
	parser.add_argument("--hidden_dim", type=int, default=512)
	parser.add_argument("--latent_dim", type=int, default=32)
	parser.add_argument("--nhead", type=int, default=8)
	parser.add_argument("--num_decoder_layers", type=int, default=2)
	parser.add_argument("--kl_beta", type=float, default=0.01)
	parser.add_argument("--kl_warmup_epochs", type=int, default=50)
	parser.add_argument("--kl_ramp_epochs", type=int, default=50)
	parser.add_argument("--image_shift_px", type=int, default=4)
	parser.add_argument("--ensemble_decay", type=float, default=0.01)
	args = parser.parse_args()

	return TrainConfig(
		dataset_id=args.dataset_id,
		split=args.split,
		output_dir=args.output_dir,
		seed=args.seed,
		val_ratio=args.val_ratio,
		epochs=args.epochs,
		batch_size=args.batch_size,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		horizon=args.horizon,
		hidden_dim=args.hidden_dim,
		latent_dim=args.latent_dim,
		nhead=args.nhead,
		num_decoder_layers=args.num_decoder_layers,
		kl_beta=args.kl_beta,
		kl_warmup_epochs=args.kl_warmup_epochs,
		kl_ramp_epochs=args.kl_ramp_epochs,
		image_shift_px=args.image_shift_px,
		ensemble_decay=args.ensemble_decay,
	)


if __name__ == "__main__":
	train(parse_args())
