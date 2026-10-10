# Flappy Bird: From Tabular to Deep Q-Learning

Follow this repository to learn reinforcement learning from first principles. Start with simple tabular Q-learning, learn about the Flappy Bird environment, and finish by training a deep Q-network (DQN) to play Flappy Bird using image pixels. 

Project: [Reinforcement Learning](https://github.com/weifengsiew/flappy-bird-deep-q-learning)

## Learning to Fly

### Before training

<img src="03_deep_q_learning/assets/gifs/before_training.gif" alt="Flappy Bird before training" width="210">

- Episode reward: `0.72`
- Average pipes successfully cleared per game (20 games): `0.0`

### After training

<img src="03_deep_q_learning/assets/gifs/after_training.gif" alt="Flappy Bird after training" width="210">

- Episode reward: `149.12`
- Average pipes successfully cleared per game (20 games): `22.8`

### Before vs after training

![Distribution of pipes successfully cleared before and after training](03_deep_q_learning/assets/plots/pipes_cleared_distribution.png)

## Learning path

### Stage 1 — Tabular Q-learning

Learn the reinforcement-learning loop—agents, environments, states, actions, rewards, returns, policies, value functions, Markov decision processes (MDPs), Bellman reasoning, exploration, and Q-learning updates—in a small hallway environment.

- Hands-on notebook: [`01_tabular_q_learning/01_tabular_q_learning.ipynb`](01_tabular_q_learning/01_tabular_q_learning.ipynb)
- Theory: [`01_tabular_q_learning/theory.md`](01_tabular_q_learning/theory.md)
- Quiz questions: [`01_tabular_q_learning/quiz_questions.md`](01_tabular_q_learning/quiz_questions.md)
- Quiz answers: [`01_tabular_q_learning/quiz_answers.md`](01_tabular_q_learning/quiz_answers.md)

### Stage 2 — Understanding visual and temporal data

Use a lightweight Flappy Bird-style environment to understand video-like observations as sequences of image frames. Focus on how pixels encode visual information over time: transform each `84 × 84` RGB frame into a normalized `42 × 42` grayscale image and stack four consecutive frames so a neural network can infer motion and temporal context.

- Hands-on notebook: [`02_intro_to_flappy_bird/02_intro_to_flappy_bird.ipynb`](02_intro_to_flappy_bird/02_intro_to_flappy_bird.ipynb)
- Theory: [`02_intro_to_flappy_bird/theory.md`](02_intro_to_flappy_bird/theory.md)
- Quiz questions: [`02_intro_to_flappy_bird/quiz_questions.md`](02_intro_to_flappy_bird/quiz_questions.md)
- Quiz answers: [`02_intro_to_flappy_bird/quiz_answers.md`](02_intro_to_flappy_bird/quiz_answers.md)

### Stage 3 — Deep Q-learning 

Replace the Q-table with a deep Q-network (DQN). Its convolutional neural network (CNN) reads the frame stack and estimates the Q-values of `do nothing` and `flap`.

- Hands-on notebook: [`03_deep_q_learning/03_deep_q_learning.ipynb`](03_deep_q_learning/03_deep_q_learning.ipynb)
- Theory: [`03_deep_q_learning/theory.md`](03_deep_q_learning/theory.md)
- Quiz questions: [`03_deep_q_learning/quiz_questions.md`](03_deep_q_learning/quiz_questions.md)
- Quiz answers: [`03_deep_q_learning/quiz_answers.md`](03_deep_q_learning/quiz_answers.md)
- Training script: [`03_deep_q_learning/train_deep_q_network.py`](03_deep_q_learning/train_deep_q_network.py)
- Training configuration: [`03_deep_q_learning/config.json`](03_deep_q_learning/config.json)
- Training script guide: [`03_deep_q_learning/03_training_guide.md`](03_deep_q_learning/03_training_guide.md)

### Stage 4 — Refactoring the DQN code

Clean up the Stage 3 training script so it is easier to understand without changing how the agent behaves.

- Split large functions into smaller ones.
- Rename unclear variables.
- Add comments explaining the training loop.
- Remove duplicated code.
- Compare the refactored results with Stage 3.

## Quick start

Use Python 3.9 or newer and install the learning dependencies:

```bash
python -m venv flappy_bird
source flappy_bird/bin/activate
python -m pip install -r requirements.txt
```

Then work through the notebooks in order. To run the final training script from the repository root:

```bash
python 03_deep_q_learning/train_deep_q_network.py
```

The script reads its hyperparameters from `03_deep_q_learning/config.json`, evaluates the policy repeatedly, and saves `deep_q_network_checkpoint.pt` and `training_metrics.json`. The metrics file contains per-episode history and evaluation checkpoints; the script also writes `assets/plots/training_progress.png` and the comparison GIFs to `assets/gifs/`.
