"""Collect targeted human teleoperation data in PushT

Usage:
python scripts/target_data_collection.py \
    --output_dir "data/data_collection" \
    --save_images=true \
"""

import os
import sys
import warnings
import math
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

flags.DEFINE_string("output_dir", "data/target", "Output directory for collected data")
flags.DEFINE_integer("start_seed", 0, "Starting environment seed (ignored if --seeds is provided)")
flags.DEFINE_integer("num_seeds", 10, "Number of seeds to collect (ignored if --seeds is provided)")
flags.DEFINE_string("seeds", None, "Comma-separated list of specific seeds (overrides start_seed/num_seeds)")
flags.DEFINE_bool("random_seeds", True, "Sample random environment seeds when --seeds is not provided")
flags.DEFINE_integer("fps", 10, "Control/render frequency in Hz")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor (>= 1.0)")
flags.DEFINE_integer("max_steps", 300, "Maximum steps per episode")
flags.DEFINE_bool("save_images", True, "Save image observations")
flags.DEFINE_float("activation_radius", 30.0, "Mouse proximity threshold for control activation")
flags.DEFINE_bool("recovery_mode", True, "Spawn T near target but misaligned for recovery data collection")
flags.DEFINE_integer("recovery_target_episodes", 20, "Successful recovery episodes to save when recovery_mode=true")
flags.DEFINE_float("recovery_spawn_std_px", 55.0, "Gaussian std (px) for T spawn around goal")
flags.DEFINE_float("recovery_min_dist_px", 15.0, "Minimum T-goal center distance (px) for recovery spawn")
flags.DEFINE_float("recovery_max_dist_px", 95.0, "Maximum T-goal center distance (px) for recovery spawn")
flags.DEFINE_float("recovery_min_angle_delta_deg", 25.0, "Minimum |angle delta| from goal orientation (deg)")
flags.DEFINE_float("recovery_max_angle_delta_deg", 170.0, "Maximum |angle delta| from goal orientation (deg)")
flags.DEFINE_integer("recovery_sample_budget", 30, "Max attempts to sample a valid recovery spawn per episode")


def get_agent_pos_from_obs(obs: Dict) -> np.ndarray:
    """Extract current agent position [x, y] from observation."""
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
    if agent_pos.ndim == 1:
        return agent_pos
    return agent_pos[-1]


def get_obs_state(obs: Dict) -> np.ndarray:
    """Build observation.state with shape (2,) (latest agent position)."""
    return get_agent_pos_from_obs(obs)


def _goal_pose_from_info(info: Dict, env) -> Optional[np.ndarray]:
    if isinstance(info, dict) and "goal_pose" in info:
        goal = np.asarray(info["goal_pose"], dtype=np.float32).reshape(-1)
        if goal.size >= 3:
            return goal[:3]

    goal = getattr(getattr(env, "unwrapped", env), "goal_pose", None)
    if goal is None:
        return None
    goal_arr = np.asarray(goal, dtype=np.float32).reshape(-1)
    if goal_arr.size < 3:
        return None
    return goal_arr[:3]


def _sample_recovery_reset_state(
    agent_pos: np.ndarray,
    goal_pose: np.ndarray,
    rng: np.random.Generator,
    spawn_std_px: float,
    min_dist_px: float,
    max_dist_px: float,
    min_angle_delta_deg: float,
    max_angle_delta_deg: float,
) -> tuple[np.ndarray, float, float]:
    gx, gy, gtheta = float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2])
    min_dist = max(0.0, float(min_dist_px))
    max_dist = max(min_dist, float(max_dist_px))

    min_deg = max(0.0, float(min_angle_delta_deg))
    max_deg = max(min_deg, float(max_angle_delta_deg))

    for _ in range(200):
        dx, dy = rng.normal(loc=0.0, scale=float(spawn_std_px), size=2)
        bx = float(np.clip(gx + dx, 0.0, 512.0))
        by = float(np.clip(gy + dy, 0.0, 512.0))
        dist = float(np.hypot(bx - gx, by - gy))
        if dist < min_dist or dist > max_dist:
            continue

        delta_deg = float(rng.uniform(min_deg, max_deg))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        btheta = float(gtheta + sign * math.radians(delta_deg))

        reset_state = np.array([agent_pos[0], agent_pos[1], bx, by, btheta], dtype=np.float32)
        return reset_state, dist, delta_deg

    # Fallback deterministic sample near boundary if Gaussian rejects repeatedly.
    fallback_theta = float(gtheta + math.radians(max(min_deg, 45.0)))
    fallback_state = np.array(
        [agent_pos[0], agent_pos[1], np.clip(gx + max_dist, 0.0, 512.0), gy, fallback_theta],
        dtype=np.float32,
    )
    fallback_dist = float(np.hypot(fallback_state[2] - gx, fallback_state[3] - gy))
    fallback_delta = abs(float(math.degrees(fallback_theta - gtheta)))
    return fallback_state, fallback_dist, fallback_delta


def _reset_episode_start(env, env_seed: int, rng: np.random.Generator, recovery_mode: bool):
    obs, info = env.reset(seed=env_seed)
    meta = {
        "recovery_requested": recovery_mode,
        "recovery_applied": False,
        "spawn_dist_px": None,
        "spawn_angle_delta_deg": None,
        "spawn_budget_exhausted": False,
        "fatal_error": None,
    }
    if not recovery_mode:
        return obs, info, meta

    goal_pose = _goal_pose_from_info(info, env)
    if goal_pose is None:
        meta["spawn_budget_exhausted"] = True
        meta["fatal_error"] = "Goal pose is unavailable from environment reset info/attributes"
        return obs, info, meta

    agent_pos = get_agent_pos_from_obs(obs)
    budget = max(1, int(FLAGS.recovery_sample_budget))
    for _ in range(budget):
        reset_state, dist, delta_deg = _sample_recovery_reset_state(
            agent_pos=agent_pos,
            goal_pose=goal_pose,
            rng=rng,
            spawn_std_px=FLAGS.recovery_spawn_std_px,
            min_dist_px=FLAGS.recovery_min_dist_px,
            max_dist_px=FLAGS.recovery_max_dist_px,
            min_angle_delta_deg=FLAGS.recovery_min_angle_delta_deg,
            max_angle_delta_deg=FLAGS.recovery_max_angle_delta_deg,
        )
        try:
            obs2, info2 = env.reset(seed=env_seed, options={"reset_to_state": reset_state.tolist()})
        except TypeError:
            meta["spawn_budget_exhausted"] = True
            meta["fatal_error"] = "Environment does not support reset(options={'reset_to_state': ...})"
            return obs, info, meta
        except Exception:
            continue

        is_success_now = bool(info2.get("is_success", False)) if isinstance(info2, dict) else False
        if is_success_now:
            continue

        meta["recovery_applied"] = True
        meta["spawn_dist_px"] = dist
        meta["spawn_angle_delta_deg"] = delta_deg
        return obs2, info2, meta

    meta["spawn_budget_exhausted"] = True
    return obs, info, meta


def _sample_random_seeds(num_seeds: int, rng: np.random.Generator) -> list[int]:
    if num_seeds <= 0:
        return []
    samples = rng.choice(2**31 - 1, size=num_seeds, replace=False)
    return samples.astype(np.int64).tolist()


def run_teleop_episode(
    env,
    controller: InterventionController,
    recorder: TrajectoryRecorder,
    env_seed: int,
    trial_idx: int,
    max_steps: int,
    fps: int,
    save_images: bool,
    recovery_mode: bool,
    recovery_rng: np.random.Generator,
    step_pbar: Optional[tqdm] = None,
):
    """Run one pure teleoperation episode.

    Human control is required for all recorded steps. The user can press Q to quit.
    """
    obs, _, spawn_meta = _reset_episode_start(
        env=env,
        env_seed=env_seed,
        rng=recovery_rng,
        recovery_mode=recovery_mode,
    )
    if recovery_mode and spawn_meta["spawn_budget_exhausted"]:
        return False, False, False, False, spawn_meta

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
            return False, False, False, True, spawn_meta

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

    return terminated, truncated, success, False, spawn_meta


def main(_):
    if FLAGS.window_scale < 1.0:
        raise ValueError("window_scale must be >= 1.0")
    if FLAGS.recovery_mode and FLAGS.recovery_target_episodes <= 0:
        raise ValueError("recovery_target_episodes must be > 0 when recovery_mode=true")

    window_size = int(512 * FLAGS.window_scale)
    seed_source_rng = np.random.default_rng()

    if FLAGS.seeds:
        seed_list = [int(s.strip()) for s in FLAGS.seeds.split(",") if s.strip()]
        if len(seed_list) == 0:
            raise ValueError("--seeds was provided but no valid integers were parsed")
        seeds_str = f"{len(seed_list)} specific seeds"
    else:
        if FLAGS.random_seeds:
            seed_list = _sample_random_seeds(FLAGS.num_seeds, seed_source_rng)
            seeds_str = f"{FLAGS.num_seeds} random seeds"
        else:
            seed_list = list(range(FLAGS.start_seed, FLAGS.start_seed + FLAGS.num_seeds))
            seeds_str = f"{FLAGS.start_seed} to {FLAGS.start_seed + FLAGS.num_seeds - 1}"

    print("=" * 60)
    print("Pure Teleoperation Data Collection")
    print("=" * 60)
    print(f"Output dir: {FLAGS.output_dir}")
    print(f"Seeds: {seeds_str}")
    print(f"Random seed mode: {FLAGS.random_seeds}")
    print(f"FPS: {FLAGS.fps}, Window: {window_size}x{window_size}")
    print(f"Recovery mode: {FLAGS.recovery_mode}")
    if FLAGS.recovery_mode:
        print(
            "Recovery spawn config: "
            f"target_success={FLAGS.recovery_target_episodes}, "
            f"std={FLAGS.recovery_spawn_std_px:.1f}px, "
            f"dist=[{FLAGS.recovery_min_dist_px:.1f}, {FLAGS.recovery_max_dist_px:.1f}]px, "
            f"angle_delta=[{FLAGS.recovery_min_angle_delta_deg:.1f}, {FLAGS.recovery_max_angle_delta_deg:.1f}]deg, "
            f"budget={FLAGS.recovery_sample_budget}"
        )
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

    total_progress = FLAGS.recovery_target_episodes if FLAGS.recovery_mode else len(seed_list)
    progress_desc = "SavedSuccess" if FLAGS.recovery_mode else "Seeds"
    seed_pbar = tqdm(total=total_progress, desc=progress_desc, position=0)
    step_pbar = tqdm(total=FLAGS.max_steps, desc="Steps", position=1, leave=False)

    recovery_rng = np.random.default_rng(int(FLAGS.start_seed) + 2026)

    saved_success_count = 0
    skipped_failure_count = 0
    skipped_spawn_count = 0

    if FLAGS.recovery_mode:
        attempt_idx = 0
        seed_cursor = 0
        while saved_success_count < FLAGS.recovery_target_episodes:
            if FLAGS.seeds:
                current_seed = seed_list[seed_cursor % len(seed_list)]
                seed_cursor += 1
            else:
                if FLAGS.random_seeds:
                    current_seed = int(seed_source_rng.integers(0, 2**31 - 1))
                else:
                    current_seed = FLAGS.start_seed + attempt_idx
            trial_idx = attempt_idx

            step_pbar.set_description(f"Seed {current_seed} T{trial_idx}")

            terminated, truncated, success, quit_requested, spawn_meta = run_teleop_episode(
                env=env,
                controller=controller,
                recorder=recorder,
                env_seed=current_seed,
                trial_idx=trial_idx,
                max_steps=FLAGS.max_steps,
                fps=FLAGS.fps,
                save_images=FLAGS.save_images,
                recovery_mode=True,
                recovery_rng=recovery_rng,
                step_pbar=step_pbar,
            )
            attempt_idx += 1

            if quit_requested:
                step_pbar.close()
                seed_pbar.close()
                print("\nQuit requested. Exiting...")
                print(f"Saved successful episodes: {saved_success_count}")
                print(f"Skipped failed episodes: {skipped_failure_count}")
                print(f"Skipped spawn-budget episodes: {skipped_spawn_count}")
                env.close()
                return

            if spawn_meta["spawn_budget_exhausted"]:
                if spawn_meta["fatal_error"] is not None:
                    step_pbar.close()
                    seed_pbar.close()
                    env.close()
                    raise RuntimeError(f"Recovery mode setup failed: {spawn_meta['fatal_error']}")
                skipped_spawn_count += 1
                seed_pbar.set_postfix_str("Last: SPAWN_SKIP")
                continue

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
                seed_pbar.update(1)
                seed_pbar.set_postfix_str(
                    "Last: SUCCESS "
                    f"d={spawn_meta['spawn_dist_px']:.1f}px "
                    f"a={spawn_meta['spawn_angle_delta_deg']:.1f}deg"
                )
            else:
                skipped_failure_count += 1
                seed_pbar.set_postfix_str("Last: FAIL")
    else:
        trial_idx = 0
        for current_seed in seed_list:
            step_pbar.set_description(f"Seed {current_seed} T{trial_idx}")

            terminated, truncated, success, quit_requested, _ = run_teleop_episode(
                env=env,
                controller=controller,
                recorder=recorder,
                env_seed=current_seed,
                trial_idx=trial_idx,
                max_steps=FLAGS.max_steps,
                fps=FLAGS.fps,
                save_images=FLAGS.save_images,
                recovery_mode=False,
                recovery_rng=recovery_rng,
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
            trial_idx += 1

    step_pbar.close()
    seed_pbar.close()

    if FLAGS.recovery_mode:
        print("\nRecovery collection complete!")
        print(f"Target successful episodes: {FLAGS.recovery_target_episodes}")
        print(f"Saved successful episodes: {saved_success_count}")
        print(f"Skipped failed episodes: {skipped_failure_count}")
        print(f"Skipped spawn-budget episodes: {skipped_spawn_count}")
    else:
        print(f"\nCollection complete! Seeds processed: {len(seed_list)}")
        print(f"Saved successful episodes: {saved_success_count}")
        print(f"Skipped failed episodes: {skipped_failure_count}")
    print(f"Output: {FLAGS.output_dir}")
    env.close()


if __name__ == "__main__":
    app.run(main)
