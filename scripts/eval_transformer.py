"""Evaluate the Hybrid Vision-Motor Transformer on the custom PushT Env."""

import os
import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import gymnasium as gym
from gymnasium import spaces
import pygame
import pymunk
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import skimage.transform as st

# ==========================================
# 1. Custom PushT Environment
# ==========================================
def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

class PushTEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self, legacy=False, block_cog=None, damping=None, render_action=True, render_size=96, reset_to_state=None):
        self._seed = None
        self.n_contact_points = 0
        self.seed()
        self.window_size = ws = 512  
        self.render_size = render_size
        self.sim_hz = 100
        self.k_p, self.k_v = 100, 20    
        self.control_hz = self.metadata['video.frames_per_second']
        self.legacy = legacy

        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,), dtype=np.float64
        )

        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,), dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        self.space = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=self._seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)
        return self._get_obs(), self._get_info()

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt
                self.space.step(dt)

        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold
        
        return self._get_obs(), reward, done, done, self._get_info()

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array' or mode is None:
            return self._render_frame()
        return None

    def _render_frame(self):
        from skimage.draw import polygon, disk
        img = np.ones((self.render_size, self.render_size, 3), dtype=np.uint8) * 255
        scale = self.render_size / self.window_size
        
        def to_px(p):
            x = int(np.clip(p[0] * scale, 0, self.render_size - 1))
            y = int(np.clip(p[1] * scale, 0, self.render_size - 1))
            return np.array([x, y], dtype=np.int32)
        
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            verts = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            pts = np.array([to_px(v) for v in verts])
            rr, cc = polygon(pts[:, 1], pts[:, 0], shape=(self.render_size, self.render_size))
            img[rr, cc] = self.goal_color
        
        for shape in self.block.shapes:
            verts = [self.block.local_to_world(v) for v in shape.get_vertices()]
            pts = np.array([to_px(v) for v in verts])
            rr, cc = polygon(pts[:, 1], pts[:, 0], shape=(self.render_size, self.render_size))
            img[rr, cc] = (119, 136, 153)  
            
        agent_pos_img = to_px(self.agent.position)
        agent_radius = max(1, int(15 * scale))
        rr, cc = disk((agent_pos_img[1], agent_pos_img[0]), agent_radius, shape=(self.render_size, self.render_size))
        img[rr, cc] = (65, 105, 225) 
        
        img = np.flipud(img)
        return img

    def _get_obs(self):
        return np.array(tuple(self.agent.position) + tuple(self.block.position) + (self.block.angle % (2 * np.pi),))

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        return {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step
        }

    def close(self): pass

    def seed(self, seed=None):
        if seed is None: seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _set_state(self, state):
        if isinstance(state, np.ndarray): state = state.tolist()
        self.agent.position = state[:2]
        self.block.angle = state[4]
        self.block.position = state[2:4]
        self.space.step(1.0 / self.sim_hz)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = (144, 238, 144) 
        self.goal_pose = np.array([256,256,np.pi/4]) 

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = (200, 200, 200)
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale), (length*scale/2, scale), (length*scale/2, 0), (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale), (-scale/2, length*scale), (scale/2, length*scale), (scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

# Register the environment
gym.envs.registration.register(id='PushT-Custom-v0', entry_point=PushTEnv, max_episode_steps=300)

# ==========================================
# 2. Hybrid Model Architecture
# ==========================================
@dataclass
class TrainConfig:
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 3
    future_steps: int = 8

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, start_idx : start_idx + seq_len, :]

class HybridVisionTransformer(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, config: TrainConfig = None):
        super().__init__()
        self.config = config
        resnet = models.resnet18(weights=None)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        vision_feature_dim = 512 
        
        self.state_encoder = nn.Linear(state_dim, 64)
        self.scene_proj = nn.Linear(vision_feature_dim + 64, config.d_model)
        
        self.action_embed = nn.Linear(action_dim, config.d_model)
        self.pe = PositionalEncoding(config.d_model)
        
        self.transformer = nn.Transformer(
            d_model=config.d_model, nhead=config.nhead, 
            num_encoder_layers=config.num_layers, num_decoder_layers=config.num_layers,
            dim_feedforward=config.d_model * 4, batch_first=True
        )
        self.action_head = nn.Linear(config.d_model, action_dim)

    def forward(self, image, state, target_actions=None):
        B = image.size(0)
        device = image.device
        
        img_features = self.vision_backbone(image.squeeze(1)).view(B, -1)
        current_state = state[:, 0, :] 
        state_features = F.relu(self.state_encoder(current_state))
        
        scene_combined = torch.cat([img_features, state_features], dim=-1)
        scene_token = self.scene_proj(scene_combined).unsqueeze(1)
        memory = self.transformer.encoder(scene_token)

        # INFERENCE MODE ONLY
        current_pos = state # (Batch, 1, 2)
        dec_in_emb = self.pe(self.action_embed(current_pos), start_idx=0)
        generated_actions = []
        
        for step in range(self.config.future_steps):
            seq_len = dec_in_emb.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            out = self.transformer.decoder(dec_in_emb, memory, tgt_mask=causal_mask)
            next_action = self.action_head(out[:, -1:, :]) 
            generated_actions.append(next_action)
            next_emb = self.pe(self.action_embed(next_action), start_idx=seq_len)
            dec_in_emb = torch.cat([dec_in_emb, next_emb], dim=1)
            
        return torch.cat(generated_actions, dim=1) 

# ==========================================
# 3. Main Evaluation Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/hybrid_transformer/best.pt")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model from {args.model_path} onto {device}...")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    config_dict = checkpoint.get("config", {})
    
    # Load hyperparams from config
    config = TrainConfig(
        d_model=config_dict.get("d_model", 256),
        nhead=config_dict.get("nhead", 4),
        num_layers=config_dict.get("num_layers", 3),
        future_steps=config_dict.get("future_steps", 8)
    )

    model = HybridVisionTransformer(config=config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Normalization Stats
    s_mean = torch.tensor(checkpoint["state_mean"], dtype=torch.float32).to(device)
    s_std = torch.tensor(checkpoint["state_std"], dtype=torch.float32).to(device)
    a_mean = torch.tensor(checkpoint["action_mean"], dtype=torch.float32).to(device)
    a_std = torch.tensor(checkpoint["action_std"], dtype=torch.float32).to(device)

    # Transforms
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((96, 96), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    env = gym.make('PushT-Custom-v0')
    
    # Setup Pygame for rendering
    pygame.init()
    window_size = env.unwrapped.window_size
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("PushT Hybrid Transformer Eval")
    clock = pygame.time.Clock()

    print("\nStarting Evaluation...")
    success_count = 0

    for seed in range(args.num_seeds):
        obs, _ = env.reset(seed=seed)
        step = 0
        terminated = False
        success = False

        while not terminated and step < args.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    env.close()
                    pygame.quit()
                    return

            # 1. Prepare Inputs
            agent_pos = obs[:2] # State is just X, Y
            img_array = env.render('rgb_array') # Shape: (96, 96, 3)

            img_tensor = base_transform(img_array).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 3, 96, 96)
            state_tensor = torch.tensor(agent_pos, dtype=torch.float32).view(1, 1, 2).to(device)
            state_tensor_norm = (state_tensor - s_mean) / s_std

            # 2. Predict Chunk
            with torch.no_grad():
                action_chunk_norm = model(img_tensor, state_tensor_norm)
            
            action_chunk = (action_chunk_norm * a_std) + a_mean
            actions = action_chunk.squeeze(0).cpu().numpy() # Shape: (8, 2)

            # 3. Execute the Action Chunk (Action Chunking in practice)
            for act in actions:
                obs, reward, terminated, truncated, info = env.step(act)
                step += 1
                
                if reward >= 1.0:
                    success = True
                    terminated = True

                # Render Step to Pygame
                render_img = env.render('rgb_array')
                render_surface = pygame.surfarray.make_surface(np.transpose(render_img, (1, 0, 2)))
                scaled_surface = pygame.transform.scale(render_surface, (window_size, window_size))
                screen.blit(scaled_surface, (0, 0))
                pygame.display.flip()
                clock.tick(10) # 10 FPS
                
                if terminated or step >= args.max_steps:
                    break

        if success:
            success_count += 1
            print(f"Episode {seed + 1}/{args.num_seeds} - SUCCESS (Took {step} steps)")
        else:
            print(f"Episode {seed + 1}/{args.num_seeds} - FAILED (Reached {step} steps)")

    print("=" * 60)
    print(f"Evaluation Complete! Success Rate: {success_count}/{args.num_seeds} ({(success_count/args.num_seeds)*100:.1f}%)")
    print("=" * 60)
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()