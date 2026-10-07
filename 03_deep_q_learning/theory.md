# Reinforcement Learning — Theory

## Concepts and checkpoints

**Stage 3 — Deep Q-learning from visual observations**

This stage trains a deep Q-network from visual frame stacks and records the improvement from random behaviour to learned behaviour.

### 1. CNN output and actions

The CNN receives a stack with shape `(4, 42, 42)` and returns two Q-values:

```text
Q(observation) = [Q(do nothing), Q(flap)]
```

Numerical example:

```text
Q(observation) = [1.4, 2.1]
greedy action = flap, because 2.1 > 1.4
```

### 2. Replay buffer

The replay buffer stores image transitions:

```text
(frame_stack, action, reward, next_frame_stack, done)
```

Random mini-batches reduce sequential correlation and let the CNN reuse past experiences.

### 3. DQN target and update

For a non-terminal transition:

```text
target = reward + γ × max(next Q-values)
```

Numerical example:

```text
reward = 0.1
γ = 0.95
next Q-values = [1.2, 2.0]
target = 0.1 + 0.95 × 2.0 = 2.0
```

### 4. Target network

The target network is a delayed copy of the policy CNN. It creates targets that change more slowly than the network being trained.

### 5. Exploration schedule

Early training uses a large ε so the bird tries both actions. Later training uses a small ε so the bird exploits its learned timing.

Numerical example:

```text
ε = 0.10
10% of actions are random
90% of actions use the highest predicted Q-value
```

### 6. Training budget and metrics

Start with `100,000` environment steps and track episode reward, pipes passed, episode length, ε, and loss. Use repeated evaluation episodes because Flappy Bird outcomes are noisy.

### 7. Before-and-after recording

Use the same game seed set, frame preprocessing, episode limit, and recording settings for both videos.

```text
before_training.gif = random or untrained agent
after_training.gif  = trained CNN-DQN with ε = 0
```

### 8. Stage 3 checkpoint

The project is successful when the trained CNN-DQN survives longer and passes more pipes than the random baseline across repeated evaluation episodes, with both recordings saved.
