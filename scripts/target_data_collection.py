# Use Python 3.11 for this whole thing, suggest creating a venv or conda env with python=3.11
# need pymunk version < 7.0.0 for gym_pusht compatibility
# pip uninstall -y pymunk && pip install "pymunk<7"
"""Collect TARGETED human teleoperation data for "last-mile" PushT recovery.

This script spawns the T-block and the agent very close to the target
with small positional and rotational offsets.

Usage:
python scripts/target_data_collection.py \
    --output_dir data/target_collection \
    --num_seeds 50 \
    --target_offset_px 3.0 \
    --target_offset_deg 10.0
"""

import os
import sys
import warnings
from typing import Dict, Optional

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import gymnasium as gym
import pygame
from absl import app, flags
from tqdm import tqdm

import gym_pusht  # noqa: F401 (registers environment)
from envs.interactive_utils import (
    ControlState,
    InterventionController,
    get_observation_image,
    draw_status_overlay,
)
from data.trajectory_recorder import TrajectoryRecorder
from data.episode_saver import EpisodeSaver

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "./teleop_target_data", "Output directory for collected data")
flags.DEFINE_integer("start_seed", 0, "Starting environment seed")
flags.DEFINE_integer("num_seeds", 10, "Number of seeds to collect")
flags.DEFINE_string("seeds", None, "Comma-separated list of specific seeds")
flags.DEFINE_integer("fps", 10, "Control/render frequency in Hz")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor (>= 1.0)")
flags.DEFINE_integer("max_steps", 100, "Maximum steps per episode (shorter for target collection)")
flags.DEFINE_bool("save_images", True, "Save image observations")
flags.DEFINE_float("activation_radius", 30.0, "Mouse proximity threshold for control activation")
flags.DEFINE_bool("random_seeds", True, "Use completely random seeds instead of sequential ones")

# --- NEW FLAGS FOR TARGETED SPAWNING ---
flags.DEFINE_float("target_offset_px", 15.0, "Max pixel distance to offset the block from the target")
flags.DEFINE_float("target_offset_deg", 10.0, "Max rotation degrees to offset the block from the target")


def override_env_state_near_target(env, offset_px: float, offset_deg: float) -> Dict:
    """Teleports the block and agent near the goal pose for recovery training."""
    unwrapped = env.unwrapped
    
    # 1. Get the goal pose. gym_pusht typically uses [256.0, 256.0, pi/4]
    gx, gy, gangle = getattr(unwrapped, 'goal_pose', (256.0, 256.0, np.pi / 4.0))
    
    # 2. Add random noise within the specified ranges
    dx = np.random.uniform(-offset_px, offset_px)
    dy = np.random.uniform(-offset_px, offset_px)
    dangle = np.random.uniform(np.radians(-offset_deg), np.radians(offset_deg))
    
    new_x = gx + dx
    new_y = gy + dy
    new_angle = gangle + dangle
    
    # 3. Apply the new position to the block (pymunk body)
    # gym_pusht exposes this either as .block or .block_body depending on the fork
    block = getattr(unwrapped, 'block', getattr(unwrapped, 'block_body', None))
    if block is not None:
        block.position = (new_x, new_y)
        block.angle = new_angle
        block.velocity = (0, 0)
        block.angular_velocity = 0
        
    # 4. Move the agent (end-effector) near the block so the human doesn't have 
    # to drag the mouse from the center of the screen
    agent_x = new_x - 20.0
    agent_y = new_y - 20.0
    agent = getattr(unwrapped, 'agent', getattr(unwrapped, 'agent_body', None))
    if agent is not None:
        agent.position = (agent_x, agent_y)
        agent.velocity = (0, 0)
        
    # 5. Take a zero-movement step to populate the observation dictionary with the new visual
    obs, _, _, _, _ = env.step(np.array([agent_x, agent_y], dtype=np.float32))
    return obs


def get_agent_pos_from_obs(obs: Dict) -> np.ndarray:
    """Extract current agent position [x, y] from observation."""
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
    if agent_pos.ndim == 1:
        return agent_pos
    return agent_pos[-1]


def get_obs_state(obs: Dict) -> np.ndarray:
    """Build observation.state with shape (2,) (latest agent position)."""
    return get_agent_pos_from_obs(obs)


def run_teleop_episode(
    env,
    controller: InterventionController,
    recorder: TrajectoryRecorder,
    env_seed: int,
    trial_idx: int,
    max_steps: int,
    fps: int,
    save_images: bool,
    step_pbar: Optional[tqdm] = None,
):
    obs, _ = env.reset(seed=env_seed)
    
    # --- TARGETED OVERRIDE HERE ---
    obs = override_env_state_near_target(env, FLAGS.target_offset_px, FLAGS.target_offset_deg)

    controller.reset()
    controller.state = ControlState.PAUSED
    recorder.reset()

    if step_pbar is not None:
        step_pbar.reset()

    step = 0
    terminated = False
    truncated = False
    success = False
    clock = pygame.time.Clock()

    while not (terminated or truncated):
        events = controller.handle_events()
        if events["quit"]:
            return False, False, False, True

        agent_pos = get_agent_pos_from_obs(obs)

        if controller.state == ControlState.PAUSED:
            controller.try_activate_human_control(agent_pos)

        if controller.state != ControlState.HUMAN_CONTROL:
            env.render()
            draw_status_overlay(env, controller.state, env_seed, trial_idx, step, max_steps, agent_pos, True)
            clock.tick(fps)
            continue

        action = controller.get_human_action(agent_pos)
        if action is None:
            env.render()
            draw_status_overlay(env, controller.state, env_seed, trial_idx, step, max_steps, agent_pos, True)
            clock.tick(fps)
            continue

        obs_state = get_obs_state(obs)
        image = get_observation_image(env) if save_images else None

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_success = bool(info.get("is_success", terminated)) if isinstance(info, dict) else bool(terminated)
        success = success or step_success

        recorder.record_step(
            obs_state=obs_state,
            raw_action=action,
            reward=reward,
            done=done,
            success=step_success,
            is_human=True,
            image=image,
        )

        step += 1
        if step_pbar is not None:
            step_pbar.update(1)

        if step >= max_steps:
            truncated = True

        env.render()
        draw_status_overlay(env, controller.state, env_seed, trial_idx, step, max_steps, agent_pos, True)
        clock.tick(fps)

    return terminated, truncated, success, False


def main(_):
    if FLAGS.window_scale < 1.0:
        raise ValueError("window_scale must be >= 1.0")

    window_size = int(512 * FLAGS.window_scale)

    # --- NEW SEED GENERATION LOGIC ---
    if FLAGS.seeds:
        seed_list = [int(s.strip()) for s in FLAGS.seeds.split(",") if s.strip()]
        seeds_str = f"{len(seed_list)} specific seeds"
    elif FLAGS.random_seeds:
        # Generate random seeds between 0 and 1,000,000
        seed_list = np.random.randint(0, 1000000, size=FLAGS.num_seeds).tolist()
        seeds_str = f"{FLAGS.num_seeds} random seeds"
    else:
        seed_list = list(range(FLAGS.start_seed, FLAGS.start_seed + FLAGS.num_seeds))
        seeds_str = f"{FLAGS.start_seed} to {FLAGS.start_seed + FLAGS.num_seeds - 1}"

    print("=" * 60)
    print("TARGETED Teleoperation Data Collection (Recovery Training)")
    print("=" * 60)
    print(f"Output dir: {FLAGS.output_dir}")
    print(f"Seeds: {seeds_str}")
    print(f"FPS: {FLAGS.fps}, Window: {window_size}x{window_size}")
    print(f"Target Offset: +/- {FLAGS.target_offset_px} px, +/- {FLAGS.target_offset_deg} deg")
    print("=" * 60)

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="human",
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    controller = InterventionController(
        activation_radius=FLAGS.activation_radius,
        window_scale=FLAGS.window_scale,
    )
    recorder = TrajectoryRecorder()
    saver = EpisodeSaver(FLAGS.output_dir)

    print("\nControls: Q=quit, move mouse near agent to control\n")

    seed_pbar = tqdm(total=len(seed_list), desc="Seeds", position=0)
    step_pbar = tqdm(total=FLAGS.max_steps, desc="Steps", position=1, leave=False)

    trial_idx = 0
    saved_success_count = 0
    skipped_failure_count = 0
    for current_seed in seed_list:
        step_pbar.set_description(f"Seed {current_seed} T{trial_idx}")

        terminated, truncated, success, quit_requested = run_teleop_episode(
            env=env,
            controller=controller,
            recorder=recorder,
            env_seed=current_seed,
            trial_idx=trial_idx,
            max_steps=FLAGS.max_steps,
            fps=FLAGS.fps,
            save_images=FLAGS.save_images,
            step_pbar=step_pbar,
        )

        if quit_requested:
            step_pbar.close()
            seed_pbar.close()
            print("\nQuit requested. Exiting...")
            print(f"Saved successful episodes: {saved_success_count}")
            print(f"Skipped failed episodes: {skipped_failure_count}")
            env.close()
            return

        if success:
            data = recorder.finalize(
                env_seed=current_seed,
                trial_idx=trial_idx,
                policy_seed=-1,
                terminated=terminated,
                truncated=truncated,
                success=success,
            )

            saver.save(
                data=data,
                images=recorder.get_images(),
                env_seed=current_seed,
                trial_idx=trial_idx,
                success=success,
                had_intervention=True,
                save_images=FLAGS.save_images,
            )
            saved_success_count += 1
            seed_pbar.set_postfix_str("Last: SUCCESS")
        else:
            skipped_failure_count += 1
            seed_pbar.set_postfix_str("Last: FAIL (skipped)")
        seed_pbar.update(1)

    step_pbar.close()
    seed_pbar.close()

    print(f"\nCollection complete! Seeds processed: {len(seed_list)}")
    print(f"Saved successful episodes: {saved_success_count}")
    print(f"Skipped failed episodes: {skipped_failure_count}")
    print(f"Output: {FLAGS.output_dir}")
    env.close()


if __name__ == "__main__":
    app.run(main)