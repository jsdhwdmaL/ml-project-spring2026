import pygame
import numpy as np
from enum import Enum

class ControlState(Enum):
    PAUSED = "PAUSED"
    HUMAN_CONTROL = "HUMAN_CONTROL"

class InterventionController:
    def __init__(self, activation_radius=30.0, window_scale=1.0):
        self.activation_radius = activation_radius
        self.window_scale = window_scale
        self.state = ControlState.PAUSED

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
        # Scale mouse position back to the default environment coordinates
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

def get_observation_image(env):
    """
    Grabs the RGB frame directly from the active pygame surface since 
    render_mode='human' often suppresses array returns.
    """
    screen = pygame.display.get_surface()
    if screen is not None:
        # Pygame surfaces are (Width, Height, RGB)
        image_array = pygame.surfarray.array3d(screen)
        # Transpose to standard (Height, Width, RGB)
        return np.transpose(image_array, (1, 0, 2))
    return np.zeros((512, 512, 3), dtype=np.uint8)

def draw_status_overlay(env, state, env_seed, trial_idx, step, max_steps, agent_pos, is_pure_teleop):
    """Draws tracking text overlay over the Push-T environment."""
    screen = pygame.display.get_surface()
    if screen is None:
        return

    pygame.font.init()
    font = pygame.font.SysFont(None, 24)
    
    text = f"Seed: {env_seed} | Trial: {trial_idx} | Step: {step}/{max_steps} | State: {state.value}"
    text_surface = font.render(text, True, (0, 0, 0))
    
    # White background for text visibility
    bg_rect = text_surface.get_rect(topleft=(10, 10))
    pygame.draw.rect(screen, (255, 255, 255), bg_rect.inflate(10, 10))
    screen.blit(text_surface, (10, 10))

    pygame.display.flip()