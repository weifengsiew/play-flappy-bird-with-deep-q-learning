# Reinforcement Learning — Learning Notes

## Concepts and checkpoints

**Stage 2 — Understanding visual and temporal data**

This stage introduces a lightweight Flappy Bird-style game as a source of video-like observations. The lesson focuses on pixel data, frame preprocessing, frame stacking, and temporal context; the agent sees rendered frames over time rather than hidden coordinates.

### 1. The action space

There are two actions:

```text
0 = do nothing
1 = flap
```

**Question:** What does action `1` do?

**Correct answer:** It flaps the bird, applying an upward impulse.

### 2. The environment interface

```text
reset() -> frame, info
step(action) -> next_frame, reward, terminated, truncated, info
```

**Question:** What does `terminated=True` mean in this game?

**Correct answer:** The bird has reached a natural terminal condition, such as colliding with a pipe or the screen boundary.

### 3. Rewards and return

The game gives a small alignment/survival reward, a larger positive reward for passing a pipe, and a negative reward for dying.

Numerical example:

```text
survival rewards: 0.1 + 0.1 + 0.1
pipe reward:      +1.0
total so far:      1.3
```

**Question:** Why give a small alignment reward before a pipe is passed?

**Correct answer:** It gives the agent a denser learning signal that encourages it to stay near the pipe gap before it successfully passes the pipe.

### 4. Rendered frames

The game produces an RGB screenshot with three colour channels.

Numerical example:

```text
RGB frame = (84, 84, 3)
number of values = 84 × 84 × 3 = 21,168
```

**Question:** What does the `3` in `(84, 84, 3)` represent?

**Correct answer:** The red, green, and blue channels of each pixel.

### 5. Preprocessing and frame stacking

Convert each RGB frame to grayscale and downsample it to `(42, 42)`. Four consecutive frames are concatenated so the CNN can infer motion.

Numerical example:

```text
one grayscale frame = (42, 42)
four-frame input = (4, 42, 42)
input values = 4 × 42 × 42 = 7,056
```

**Question:** Why use four frames instead of one?

**Correct answer:** A sequence shows whether the bird and pipes are moving and in which direction.

### 6. Pixels versus hidden coordinates

The environment internally knows the bird's position, velocity, pipe position, and gap. The final agent receives only the processed frame stack.

**Question:** Why must the agent not receive the bird's hidden coordinates?

**Correct answer:** The project is testing visual decision-making, so the agent must act using only information available in the screenshot.

### 7. Stage 2 checkpoint

The stage is complete when the game can reset, both actions work, rewards and terminal states are correct, and the CNN input always has shape `(4, 42, 42)`.

**Question:** Name two checks before training the CNN-DQN.

**Correct answer:** Verify that frames have a fixed shape and that both actions change the game correctly. Also verify reward timing, collision detection, and frame stacking.
