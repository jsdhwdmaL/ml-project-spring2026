from __future__ import annotations

import pygame
import numpy as np
from enum import Enum
from typing import Optional

class ControlState(Enum):
    PAUSED = "PAUSED"
    HUMAN_CONTROL = "HUMAN_CONTROL"
    MODEL_CONTROL = "MODEL_CONTROL"  # Added for inference
    BLENDED_INTERVENTION = "BLENDED_INTERVENTION"  # CR-DAgger style soft correction

class InterventionController:
    def __init__(
        self,
        activation_radius=30.0,
        window_scale=1.0,
        blend_lambda: Optional[float] = None,
    ):
        self.activation_radius = activation_radius
        self.window_scale = window_scale
        self.state = ControlState.PAUSED
        # blend_lambda is None -> original overwrite-only behavior (backwards compatible).
        # When set, callers may use try_activate_blended_control / get_blended_action.
        self.blend_lambda = blend_lambda

    def reset(self):
        self.state = ControlState.PAUSED

    def handle_events(self):
        """Processes keyboard and window events."""
        events = {"quit": False}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events["quit"] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    events["quit"] = True
        return events

    def try_activate_human_control(self, agent_pos):
        """Activates if the mouse gets close to the agent or clicks."""
        mouse_pos = pygame.mouse.get_pos()
        env_x = mouse_pos[0] / self.window_scale
        env_y = mouse_pos[1] / self.window_scale
        
        dist = np.linalg.norm(np.array([env_x, env_y]) - agent_pos)
        
        if dist <= self.activation_radius or pygame.mouse.get_pressed()[0]:
            self.state = ControlState.HUMAN_CONTROL

    def get_human_action(self, agent_pos):
        """Returns the mapped 2D coordinate as the environment action."""
        mouse_pos = pygame.mouse.get_pos()
        env_x = mouse_pos[0] / self.window_scale
        env_y = mouse_pos[1] / self.window_scale
        
        return np.array([env_x, env_y], dtype=np.float32)

    # ---------- CR-DAgger blended-intervention API ----------

    def is_blend_engaged(self) -> bool:
        """True iff the human is engaging blended intervention this step.

        Engagement is "either SHIFT key held". Only meaningful when
        blend_lambda is not None.
        """
        if self.blend_lambda is None:
            return False
        keys = pygame.key.get_pressed()
        return bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

    def try_activate_blended_control(self) -> None:
        """Switch to BLENDED_INTERVENTION while SHIFT is held; otherwise
        fall back to MODEL_CONTROL (released)."""
        if self.blend_lambda is None:
            return
        if self.is_blend_engaged():
            self.state = ControlState.BLENDED_INTERVENTION
        else:
            self.state = ControlState.MODEL_CONTROL

    def get_blended_action(self, a_base: np.ndarray) -> np.ndarray:
        """Return (1 - lambda) * a_base + lambda * mouse_target, clipped to [0, 512].

        a_base is the base policy's intended action for this step (2-D).
        """
        if self.blend_lambda is None:
            raise RuntimeError(
                "get_blended_action called but blend_lambda is None; "
                "construct InterventionController with blend_lambda=<float>."
            )
        lam = float(self.blend_lambda)
        mouse_pos = pygame.mouse.get_pos()
        a_human = np.array(
            [mouse_pos[0] / self.window_scale, mouse_pos[1] / self.window_scale],
            dtype=np.float32,
        )
        a_base = np.asarray(a_base, dtype=np.float32).reshape(-1)
        blended = (1.0 - lam) * a_base + lam * a_human
        return np.clip(blended, 0.0, 512.0).astype(np.float32)

def get_observation_image(env):
    """Grabs the RGB frame directly from the active pygame surface."""
    screen = pygame.display.get_surface()
    if screen is not None:
        image_array = pygame.surfarray.array3d(screen)
        return np.transpose(image_array, (1, 0, 2))
    return np.zeros((512, 512, 3), dtype=np.uint8)

def _draw_future_plan_on_pygame(
    screen: pygame.surface.Surface,
    future_plan_xy: np.ndarray,
) -> None:
    """Draw predicted action chunk (H,2) in env coordinates [0,512] on the PushT pygame surface."""
    pts = np.asarray(future_plan_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return
    xy = [tuple(p) for p in np.clip(pts, 0.0, 512.0)]
    if len(xy) > 1:
        pygame.draw.aalines(screen, (60, 220, 100), False, xy, blend=1)
    for p in xy:
        pygame.draw.circle(screen, (255, 200, 0), (int(p[0]), int(p[1])), 4, width=0)


def draw_future_plan_on_rgb_frame(frame: np.ndarray, future_plan_xy: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) — draws chunk path + label; for headless / cv2 pipeline."""
    import cv2

    out = frame.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    pts = np.asarray(future_plan_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 1:
        return out
    pi = np.clip(pts, 0.0, 512.0).astype(np.int32)
    if pi.shape[0] >= 2:
        cv2.polylines(bgr, [pi], isClosed=False, color=(100, 220, 60), thickness=2, lineType=cv2.LINE_AA)
    for x, y in pi:
        cv2.circle(bgr, (int(x), int(y)), 4, (0, 200, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(
        bgr, "Future plan (ACT chunk)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 60), 2, cv2.LINE_AA
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_status_overlay(
    env,
    state,
    env_seed,
    trial_idx,
    step,
    max_steps,
    agent_pos,
    is_pure_teleop,
    reward=None,
    future_plan_xy: Optional[np.ndarray] = None,
):
    """Draws tracking text overlay over the Push-T environment.

    future_plan_xy: optional (H,2) model chunk in pixel space [0,512] to visualize as a future path.
    """
    screen = pygame.display.get_surface()
    if screen is None:
        return

    if future_plan_xy is not None:
        _draw_future_plan_on_pygame(screen, future_plan_xy)

    # Use a default font that is likely to exist on Mac/Linux
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 20, bold=True)

    _, height = screen.get_size()

    # Text color: Red if human is interfering, Blue if model is running, Black if paused
    color = (0, 0, 0)
    if state == ControlState.HUMAN_CONTROL:
        color = (200, 0, 0)
    elif state == ControlState.MODEL_CONTROL:
        color = (0, 0, 200)
    elif state == ControlState.BLENDED_INTERVENTION:
        color = (160, 0, 200)  # purple = blended (mix of red human + blue model)

    text = f"Seed: {env_seed} | Step: {step}/{max_steps} | {state.value}"
    text_surface = font.render(text, True, color)

    # Draw a semi-transparent box or simple white background for text
    bg_rect = text_surface.get_rect(topleft=(15, 15))
    pygame.draw.rect(screen, (255, 255, 255), bg_rect.inflate(10, 10))
    screen.blit(text_surface, (15, 15))

    if reward is not None:
        r = float(reward)
        reward_text = f"{r:.4f}"
        reward_surface = font.render(reward_text, True, (0, 0, 0))
        reward_rect = reward_surface.get_rect()
        reward_rect.bottomleft = (15, height - 15)
        reward_bg = reward_rect.inflate(10, 10)
        pygame.draw.rect(screen, (255, 255, 255), reward_bg)
        screen.blit(reward_surface, reward_rect)

    pygame.display.flip()
