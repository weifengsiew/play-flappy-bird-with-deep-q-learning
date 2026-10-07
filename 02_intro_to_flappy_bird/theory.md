# Reinforcement Learning — Theory

## Concepts and checkpoints

**Stage 2 — Understanding visual and temporal data**

This stage introduces a lightweight Flappy Bird-style game as a source of video-like observations. The lesson focuses on pixel data, frame preprocessing, frame stacking, and temporal context; the agent sees rendered frames over time rather than hidden coordinates.

### 1. The action space

There are two actions:

```text
0 = do nothing
1 = flap
```

### 2. The environment interface

```text
reset() -> frame, info
step(action) -> next_frame, reward, terminated, truncated, info
```

### 3. Rewards and return

The game gives a small alignment/survival reward, a larger positive reward for passing a pipe, and a negative reward for dying.

Numerical example:

```text
survival rewards: 0.1 + 0.1 + 0.1
pipe reward:      +1.0
total so far:      1.3
```

### 4. Rendered frames

The game produces an RGB screenshot with three colour channels.

Numerical example:

```text
RGB frame = (84, 84, 3)
number of values = 84 × 84 × 3 = 21,168
```

### 5. Preprocessing and frame stacking

Convert each RGB frame to grayscale and downsample it to `(42, 42)`. Four consecutive frames are concatenated so the CNN can infer motion.

Numerical example:

```text
one grayscale frame = (42, 42)
four-frame input = (4, 42, 42)
input values = 4 × 42 × 42 = 7,056
```

### 6. Pixels versus hidden coordinates

The environment internally knows the bird's position, velocity, pipe position, and gap. The final agent receives only the processed frame stack.

### 7. Stage 2 checkpoint

The stage is complete when the game can reset, both actions work, rewards and terminal states are correct, and the CNN input always has shape `(4, 42, 42)`.
