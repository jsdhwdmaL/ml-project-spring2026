import pygame
import numpy as np
from enum import Enum

class ControlState(Enum):
    PAUSED = "PAUSED"
    HUMAN_CONTROL = "HUMAN_CONTROL"
    MODEL_CONTROL = "MODEL_CONTROL"  # Added for inference

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
    """Grabs the RGB frame directly from the active pygame surface."""
    screen = pygame.display.get_surface()
    if screen is not None:
        image_array = pygame.surfarray.array3d(screen)
        return np.transpose(image_array, (1, 0, 2))
    return np.zeros((512, 512, 3), dtype=np.uint8)

def draw_status_overlay(env, state, env_seed, trial_idx, step, max_steps, agent_pos, is_pure_teleop):
    """Draws tracking text overlay over the Push-T environment."""
    screen = pygame.display.get_surface()
    if screen is None:
        return

    # Use a default font that is likely to exist on Mac/Linux
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 20, bold=True)
    
    # Text color: Red if human is interfering, Blue if model is running, Black if paused
    color = (0, 0, 0)
    if state == ControlState.HUMAN_CONTROL:
        color = (200, 0, 0)
    elif state == ControlState.MODEL_CONTROL:
        color = (0, 0, 200)

    text = f"Seed: {env_seed} | Step: {step}/{max_steps} | {state.value}"
    text_surface = font.render(text, True, color)
    
    # Draw a semi-transparent box or simple white background for text
    bg_rect = text_surface.get_rect(topleft=(15, 15))
    pygame.draw.rect(screen, (255, 255, 255), bg_rect.inflate(10, 10))
    screen.blit(text_surface, (15, 15))

    pygame.display.flip()
