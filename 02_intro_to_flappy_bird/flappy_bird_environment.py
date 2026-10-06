"""Small, dependency-light Flappy Bird-style Gymnasium environment."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlappyBirdEnv(gym.Env):
    """An 84x84 visual game with actions 0=no-op and 1=flap."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, width=84, height=84, max_steps=1200, seed=7):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
        self.pipe_width = 10
        self.gap_height = 34

    def _spawn_pipe(self):
        self.pipe_x = float(self.width + 8)
        self.gap_y = int(self.rng.integers(18, self.height - 18))
        self.pipe_counted = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.bird_x = 18.0
        self.bird_y = self.height / 2
        self.bird_velocity = 0.0
        self.steps = 0
        self.pipes_passed = 0
        self._spawn_pipe()
        self.pipe_x = float(self.width * 0.55)
        return self.render(), {"pipes_passed": self.pipes_passed}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("action must be 0 (no-op) or 1 (flap)")

        if action == 1:
            self.bird_velocity = -3.5
        self.bird_velocity += 0.35
        self.bird_y += self.bird_velocity
        self.pipe_x -= 2.2
        self.steps += 1

        distance_to_gap = abs(self.bird_y - self.gap_y)
        normalized_distance = min(distance_to_gap, self.height / 2) / (self.height / 2)
        reward = 0.2 * (1.0 - normalized_distance)
        if not self.pipe_counted and self.pipe_x + self.pipe_width < self.bird_x:
            self.pipe_counted = True
            self.pipes_passed += 1
            reward += 1.0

        terminated = self._collision()
        if terminated:
            reward = -1.0
        elif self.pipe_x + self.pipe_width < 0:
            self._spawn_pipe()

        truncated = self.steps >= self.max_steps
        info = {"pipes_passed": self.pipes_passed}
        return self.render(), reward, terminated, truncated, info

    def _collision(self):
        bird_left = self.bird_x - 2
        bird_right = self.bird_x + 2
        bird_top = self.bird_y - 2
        bird_bottom = self.bird_y + 2
        hit_boundary = bird_top <= 0 or bird_bottom >= self.height
        overlaps_pipe = (
            self.pipe_x < bird_right and self.pipe_x + self.pipe_width > bird_left
        )
        gap_top = self.gap_y - self.gap_height / 2
        gap_bottom = self.gap_y + self.gap_height / 2
        hits_pipe = overlaps_pipe and (bird_top < gap_top or bird_bottom > gap_bottom)
        return hit_boundary or hits_pipe

    def render(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = [135, 206, 235]

        pipe_left = max(0, int(self.pipe_x))
        pipe_right = min(self.width, int(self.pipe_x + self.pipe_width))
        gap_top = int(self.gap_y - self.gap_height / 2)
        gap_bottom = int(self.gap_y + self.gap_height / 2)
        if pipe_left < pipe_right:
            frame[:gap_top, pipe_left:pipe_right] = [34, 139, 34]
            frame[gap_bottom:, pipe_left:pipe_right] = [34, 139, 34]

        bird_left = max(0, int(self.bird_x - 2))
        bird_right = min(self.width, int(self.bird_x + 3))
        bird_top = max(0, int(self.bird_y - 2))
        bird_bottom = min(self.height, int(self.bird_y + 3))
        frame[bird_top:bird_bottom, bird_left:bird_right] = [255, 215, 0]
        return frame
