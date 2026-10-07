# Reinforcement Learning Curriculum

## Final goal

Progress from tabular Q-learning to a deep Q-network (DQN) that uses a convolutional neural network (CNN) to play a lightweight Flappy Bird-style game from rendered frames, then produce before-training and after-training gameplay recordings.

The game has two actions:

```text
0 = do nothing
1 = flap
```

Training is deliberately sized for a sub-one-hour CPU experiment. The agent receives low-resolution frames, not the game's hidden coordinates.

## Learning method

Each stage uses:

1. One short concept explanation.
2. One numerical example for vectors, matrices, returns, or updates.
3. One representative checkpoint question.
4. One runnable experiment or notebook activity.
5. Separate theory, quiz-question, and quiz-answer notes for each stage.

## Stage 1 — Foundations and tabular Q-learning

### Objective

Learn the reinforcement-learning loop, value functions, Bellman reasoning, and Q-learning in a small hallway environment before working with images or neural networks.

### Materials

- Virtual environment: `flappy_bird`
- Python: 3.9.20
- Kernel: `Python (reinforcement-learning)`
- Notebook: `01_tabular_q_learning/01_tabular_q_learning.ipynb`
- Theory: `01_tabular_q_learning/theory.md`
- Quiz questions: `01_tabular_q_learning/quiz_questions.md`
- Quiz answers: `01_tabular_q_learning/quiz_answers.md`

### Concepts

- Agent, environment, state, action, reward, episode, and terminal state.
- Return, discount factor `γ`, policy, `V(s)`, and `Q(s,a)`.
- Markov decision processes.
- Exploration versus exploitation and ε-greedy selection.
- Q-table, TD error, learning rate `α`, and the Q-learning update.

### Experiment

Use the hallway environment and Q-table to learn a policy that moves the agent from the start state to the goal.

### Pass criteria

The hallway agent reliably reaches its goal, and you can explain how reward information propagates backward through earlier state-action pairs.

## Stage 2 — Understanding visual and temporal data

### Objective

Understand how video-like observations are represented as sequences of image frames. Use the Flappy Bird-style environment to study pixel data, frame shapes, preprocessing, frame stacking, and temporal context before focusing on learning from rewards.

### Materials

- Environment: `02_intro_to_flappy_bird/flappy_bird_environment.py`
- Notebook: `02_intro_to_flappy_bird/02_intro_to_flappy_bird.ipynb`
- Theory: `02_intro_to_flappy_bird/theory.md`
- Quiz questions: `02_intro_to_flappy_bird/quiz_questions.md`
- Quiz answers: `02_intro_to_flappy_bird/quiz_answers.md`

### Environment interface

```text
reset() -> frame, info
step(action) -> next_frame, reward, terminated, truncated, info
```

The bird falls under gravity. Flapping gives an upward impulse. Passing a pipe gives positive reward, alignment with the gap gives a small shaping reward, and collision gives a negative reward.

### Visual pipeline

1. Capture the RGB frame.
2. Convert it to grayscale.
3. Downsample it to `42 × 42`.
4. Stack four consecutive frames to expose motion.
5. Feed the stack to a CNN.

### Numerical example

```text
RGB frame = (84, 84, 3)
grayscale frame = (42, 42)
four-frame CNN input = (4, 42, 42)
number of input values = 4 × 42 × 42 = 7,056
```

### Experiment

Run the environment repeatedly, inspect rendered frames, and verify the preprocessing and frame-stacking pipeline before training a network.

### Pass criteria

Frames render consistently, processed observations have shape `(4, 42, 42)`, frame stacking preserves temporal information, and no hidden coordinate is passed to the agent.

## Stage 3 — Deep Q-learning from visual observations

### Objective

Replace the Q-table with a DQN whose CNN estimates two action values from a four-frame visual observation, then evaluate the learned policy against a random baseline.

### Materials

- Notebook: `03_deep_q_learning/03_deep_q_learning.ipynb`
- Training script: `03_deep_q_learning/train_deep_q_network.py`
- Theory: `03_deep_q_learning/theory.md`
- Quiz questions: `03_deep_q_learning/quiz_questions.md`
- Quiz answers: `03_deep_q_learning/quiz_answers.md`
- Guide: `03_deep_q_learning/03_training_guide.md`

### DQN loop

```text
frame stack
→ CNN
→ [Q(do nothing), Q(flap)]
→ ε-greedy action
→ game reward and next frame
→ replay buffer
→ TD update
```

### Training configuration

- Train for up to `800,000` environment steps.
- Decay `ε` from approximately `1.0` to `0.05` over the first `60,000` steps.
- Evaluate every `10,000` steps across seeds `200`–`219`.
- Stop early when the evaluation median reaches `30` pipes.
- Train headless for speed.
- Use one CPU thread and a fixed random seed for reproducibility.

### Evaluation and recordings

1. Record a random-policy baseline with environment seed `101`.
2. Train the CNN-based DQN and save its weights.
3. Set `ε=0` during evaluation.
4. Record the trained policy with the same environment seed and settings.
5. Compare the before/after rewards and repeated evaluation pipes-passed scores.

The script writes the checkpoint to `deep_q_network_checkpoint.pt`, evaluation data to `training_metrics.json`, and the two recordings to `assets/gifs/`.

### Pass criteria

The trained agent survives longer and passes more pipes than the random baseline across repeated seeds, and both gameplay recordings are saved.

## After the core goal

Study Double DQN, dueling networks, prioritized replay, reward shaping, partial observability, and recurrent DQN.

## Deliverables

- `curriculum.md`
- `01_tabular_q_learning/01_tabular_q_learning.ipynb`, `01_tabular_q_learning/theory.md`, `01_tabular_q_learning/quiz_questions.md`, and `01_tabular_q_learning/quiz_answers.md`
- `02_intro_to_flappy_bird/flappy_bird_environment.py`, `02_intro_to_flappy_bird/02_intro_to_flappy_bird.ipynb`, `02_intro_to_flappy_bird/theory.md`, `02_intro_to_flappy_bird/quiz_questions.md`, and `02_intro_to_flappy_bird/quiz_answers.md`
- `03_deep_q_learning/03_deep_q_learning.ipynb`, `03_deep_q_learning/train_deep_q_network.py`, `03_deep_q_learning/theory.md`, `03_deep_q_learning/quiz_questions.md`, `03_deep_q_learning/quiz_answers.md`, and `03_deep_q_learning/03_training_guide.md`
- `03_deep_q_learning/deep_q_network_checkpoint.pt` and `03_deep_q_learning/training_metrics.json`
- Before-training and after-training gameplay GIFs in `03_deep_q_learning/assets/gifs/`.
