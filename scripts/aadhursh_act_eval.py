"""Evaluate the aadarshram/act_pusht LeRobot-native ACT model in PushT."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torchvision.transforms as T
import gymnasium as gym
import pygame
from absl import app, flags

import gym_pusht
from lerobot.policies.act.modeling_act import ACTPolicy
from envs.interactive_utils import get_observation_image, draw_status_overlay, ControlState

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_seeds", 5, "Number of episodes to evaluate")
flags.DEFINE_boolean("random_seeds", True, "Use random seeds")
flags.DEFINE_integer("fps", 10, "Control frequency")
flags.DEFINE_integer("max_steps", 300, "Max steps per episode")
flags.DEFINE_float("window_scale", 1.0, "Window scale")
flags.DEFINE_string("model_id", "aadarshram/act_pusht", "HuggingFace repo id")


def main(_):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading {FLAGS.model_id} onto {device}...")

    policy = ACTPolicy.from_pretrained(FLAGS.model_id)
    policy = policy.to(device)
    policy.eval()

    # Image preprocessing: resize to 96x96
    preprocess = T.Compose([
        T.ToTensor(),
        T.Resize((96, 96), antialias=True),
    ])

    window_size = int(512 * FLAGS.window_scale)
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="human",
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    seeds = (
        np.random.randint(0, 2**31, size=FLAGS.num_seeds).tolist()
        if FLAGS.random_seeds
        else list(range(FLAGS.num_seeds))
    )

    success_count = 0
    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        policy.reset()  # clears temporal ensemble / action queue

        step = 0
        terminated = truncated = success = False
        clock = pygame.time.Clock()

        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q
                ):
                    env.close()
                    return

            # Extract agent pos (handle stacked or flat obs)
            agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
            if agent_pos.ndim > 1:
                agent_pos = agent_pos[-1]

            # Build image tensor  (1, 3, 96, 96)
            img_array = get_observation_image(env)
            image_tensor = preprocess(img_array).unsqueeze(0).to(device)

            # Build state tensor  (1, 2)
            state_tensor = torch.tensor(agent_pos, dtype=torch.float32, device=device).unsqueeze(0)

            # LeRobot policy expects a batch dict
            batch = {
                "observation.image": image_tensor,
                "observation.state": state_tensor,
            }

            with torch.no_grad():
                action_tensor = policy.select_action(batch)  # (1, 2) or (2,)

            action = action_tensor.squeeze().cpu().numpy().astype(np.float32)
            action = np.clip(action, 0.0, 512.0)

            obs, reward, terminated, truncated, info = env.step(action)
            step_success = bool(info.get("is_success", terminated)) if isinstance(info, dict) else bool(terminated)
            success = success or step_success
            step += 1

            env.render()
            draw_status_overlay(
                env, ControlState.MODEL_CONTROL, int(seed), 0,
                step, FLAGS.max_steps, agent_pos, False,
            )
            clock.tick(FLAGS.fps)

        if success:
            success_count += 1
        print(f"Episode {i+1}/{FLAGS.num_seeds} seed={seed} {'SUCCESS' if success else 'FAILED'} ({step} steps)")

    print("=" * 60)
    print(f"Success: {success_count}/{FLAGS.num_seeds} ({100*success_count/FLAGS.num_seeds:.1f}%)")
    print("=" * 60)
    env.close()


if __name__ == "__main__":
    app.run(main)