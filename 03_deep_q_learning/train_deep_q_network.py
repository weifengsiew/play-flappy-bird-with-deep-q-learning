"""Train a small CNN-DQN on the local Flappy Bird-style environment."""

import json
import random
import sys
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "02_intro_to_flappy_bird"))
from flappy_bird_environment import FlappyBirdEnv


SEED = 7
ACTION_COUNT = 2
OBSERVATION_SHAPE = (4, 42, 42)
TRAINING_STEPS = 800_000
EPSILON_DECAY_STEPS = 60_000
MIN_EPSILON = 0.05
EVALUATION_INTERVAL = 10_000
TARGET_MEDIAN_PIPES = 30
EVALUATION_SEEDS = range(200, 220)


def preprocess(frame):
    grayscale = frame.mean(axis=2).astype(np.float32) / 255.0
    return grayscale[::2, ::2]


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_COUNT),
        )

    def forward(self, observations):
        return self.head(self.features(observations.float()))


Transition = namedtuple("Transition", "state action reward next_state done")


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = Transition(*zip(*random.sample(self.memory, batch_size)))
        return (
            torch.tensor(np.stack(batch.state), dtype=torch.float32),
            torch.tensor(batch.action, dtype=torch.long),
            torch.tensor(batch.reward, dtype=torch.float32),
            torch.tensor(np.stack(batch.next_state), dtype=torch.float32),
            torch.tensor(batch.done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.memory)


def choose_action(network, observation, epsilon):
    if network is None or random.random() < epsilon:
        return random.randrange(ACTION_COUNT)
    with torch.no_grad():
        values = network(torch.tensor(observation).unsqueeze(0))
    return int(values.argmax(dim=1).item())


def optimize_step(policy_net, target_net, replay_buffer, optimizer, batch_size=64):
    if len(replay_buffer) < batch_size:
        return None
    states, actions, rewards, next_states, done = replay_buffer.sample(batch_size)
    current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        targets = rewards + 0.95 * next_q * (1.0 - done)
    loss = F.smooth_l1_loss(current_q, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def run_episode(env, network=None, epsilon=0.0, train=False, replay=None, optimizer=None):
    frame, info = env.reset()
    history = [preprocess(frame)] * 4
    total_reward = 0.0
    losses = []
    steps = 0
    while True:
        observation = np.stack(history)
        action = choose_action(network, observation, epsilon)
        next_frame, reward, terminated, truncated, info = env.step(action)
        next_history = history[1:] + [preprocess(next_frame)]
        next_observation = np.stack(next_history)
        if train:
            replay.add(observation, action, reward, next_observation, float(terminated or truncated))
            if steps % 4 == 0:
                loss = optimize_step(network, target_net, replay, optimizer)
                if loss is not None:
                    losses.append(loss)
        history = next_history
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    return total_reward, info["pipes_passed"], steps, losses


def record_episode(env, action_selector):
    frame, _ = env.reset()
    history = [preprocess(frame)] * 4
    frames = [frame.copy()]
    total_reward = 0.0
    while True:
        action = action_selector(np.stack(history))
        frame, reward, terminated, truncated, _ = env.step(action)
        history = history[1:] + [preprocess(frame)]
        frames.append(frame.copy())
        total_reward += reward
        if terminated or truncated:
            break
    return frames, total_reward


def evaluate_policy(policy_net, seeds):
    scores = []
    for seed in seeds:
        env = FlappyBirdEnv(seed=seed)
        _, pipes, _, _ = run_episode(env, policy_net, epsilon=0.0)
        scores.append(pipes)
        env.close()
    return scores


def save_gif(frames, path, fps=15):
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)


def main():
    global target_net
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    output_dir = Path(__file__).resolve().parent
    gif_dir = output_dir / "assets" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "deep_q_network_checkpoint.pt"
    metrics_path = output_dir / "training_metrics.json"

    policy_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    replay = ReplayBuffer()

    steps_seen = 0
    episode = 0
    if checkpoint.exists():
        policy_net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        target_net.load_state_dict(policy_net.state_dict())
        if metrics_path.exists():
            previous_metrics = json.loads(metrics_path.read_text())
            steps_seen = int(previous_metrics.get("training_steps", 0))
            episode = int(previous_metrics.get("episodes", 0))
        print(f"resuming from episode={episode} steps={steps_seen}", flush=True)

    recent_scores = deque(maxlen=50)
    next_evaluation_steps = steps_seen + EVALUATION_INTERVAL
    while steps_seen < TRAINING_STEPS:
        episode += 1
        epsilon = max(
            MIN_EPSILON,
            1.0 - (1.0 - MIN_EPSILON) * steps_seen / EPSILON_DECAY_STEPS,
        )
        env = FlappyBirdEnv(seed=SEED + episode)
        reward, pipes, steps, losses = run_episode(
            env, policy_net, epsilon, train=True, replay=replay, optimizer=optimizer
        )
        env.close()
        steps_seen += steps
        recent_scores.append(pipes)
        if steps_seen % 1_000 < steps:
            target_net.load_state_dict(policy_net.state_dict())
        if episode % 100 == 0:
            print(
                f"episode={episode:5d} steps={steps_seen:7d} epsilon={epsilon:.3f} "
                f"recent_pipes={np.mean(recent_scores):.2f} "
                f"loss={np.mean(losses) if losses else float('nan'):.4f}",
                flush=True,
            )
        if steps_seen >= next_evaluation_steps:
            evaluation_scores = evaluate_policy(policy_net, EVALUATION_SEEDS)
            evaluation_median = float(np.median(evaluation_scores))
            print(
                f"evaluation steps={steps_seen} "
                f"mean_pipes={np.mean(evaluation_scores):.2f} "
                f"median_pipes={evaluation_median:.2f}",
                flush=True,
            )
            if evaluation_median >= TARGET_MEDIAN_PIPES:
                print(
                    f"target reached: median_pipes={evaluation_median:.2f} "
                    f">= {TARGET_MEDIAN_PIPES}",
                    flush=True,
                )
                break
            next_evaluation_steps += EVALUATION_INTERVAL

    torch.save(policy_net.state_dict(), checkpoint)

    def random_selector(_):
        return random.randrange(ACTION_COUNT)

    def trained_selector(observation):
        return choose_action(policy_net, observation, epsilon=0.0)

    before_frames, before_reward = record_episode(FlappyBirdEnv(seed=101), random_selector)
    after_frames, after_reward = record_episode(FlappyBirdEnv(seed=101), trained_selector)
    save_gif(before_frames, gif_dir / "before_training.gif")
    save_gif(after_frames, gif_dir / "after_training.gif")

    scores = evaluate_policy(policy_net, EVALUATION_SEEDS)

    metrics = {
        "training_steps": steps_seen,
        "episodes": episode,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "evaluation_pipes_mean": float(np.mean(scores)),
        "evaluation_pipes_median": float(np.median(scores)),
        "evaluation_pipes": scores,
        "before_reward": before_reward,
        "after_reward": after_reward,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved checkpoint: {checkpoint}")
    print(f"evaluation mean pipes: {metrics['evaluation_pipes_mean']:.2f}")
    print(f"evaluation median pipes: {metrics['evaluation_pipes_median']:.2f}")
    print(f"baseline reward: {before_reward:.2f}")
    print(f"trained reward: {after_reward:.2f}")


if __name__ == "__main__":
    main()
