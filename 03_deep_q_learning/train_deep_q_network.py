"""Train a small CNN-DQN on the local Flappy Bird-style environment."""

import json
import random
import sys
from collections import deque, namedtuple
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "02_intro_to_flappy_bird"))
from flappy_bird_environment import FlappyBirdEnv


CONFIG_PATH = Path(__file__).resolve().with_name("config.json")
CONFIG = json.loads(CONFIG_PATH.read_text())
SEED = CONFIG["seed"]
ACTION_COUNT = CONFIG["action_count"]


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
    def __init__(self, capacity):
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


def optimize_step(policy_net, target_net, replay_buffer, optimizer, batch_size):
    if len(replay_buffer) < batch_size:
        return None
    states, actions, rewards, next_states, done = replay_buffer.sample(batch_size)
    current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        targets = rewards + CONFIG["discount_factor"] * next_q * (1.0 - done)
    loss = F.smooth_l1_loss(current_q, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(), CONFIG["gradient_clip_norm"]
    )
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
            if steps % CONFIG["optimize_every"] == 0:
                loss = optimize_step(
                    network,
                    target_net,
                    replay,
                    optimizer,
                    CONFIG["batch_size"],
                )
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


def save_gif(frames, path, fps):
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)


def save_training_plot(training_history, evaluation_history, path):
    if not training_history:
        return
    episodes = np.array([item["episode"] for item in training_history])
    rewards = np.array([item["reward"] for item in training_history])
    pipes = np.array([item["pipes"] for item in training_history])
    window = min(CONFIG["rolling_window"], len(training_history))
    rolling_kernel = np.ones(window) / window

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    axes[0].plot(episodes, rewards, alpha=0.2, color="tab:blue")
    axes[0].plot(
        episodes[window - 1:],
        np.convolve(rewards, rolling_kernel, mode="valid"),
        color="tab:blue",
        label=f"{window}-episode rolling mean",
    )
    axes[0].set_ylabel("Episode reward")
    axes[0].legend()

    axes[1].plot(episodes, pipes, alpha=0.2, color="tab:green")
    axes[1].plot(
        episodes[window - 1:],
        np.convolve(pipes, rolling_kernel, mode="valid"),
        color="tab:green",
        label=f"{window}-episode rolling mean",
    )
    axes[1].set_ylabel("Pipes passed")
    axes[1].legend()

    if evaluation_history:
        evaluation_steps = [item["training_steps"] for item in evaluation_history]
        evaluation_means = [item["mean_pipes"] for item in evaluation_history]
        evaluation_medians = [item["median_pipes"] for item in evaluation_history]
        axes[2].plot(evaluation_steps, evaluation_means, marker="o", label="Mean")
        axes[2].plot(
            evaluation_steps,
            evaluation_medians,
            marker="o",
            label="Median",
        )
        axes[2].axhline(
            CONFIG["target_median_pipes"],
            linestyle="--",
            color="tab:red",
            label="Target median",
        )
        axes[2].legend()
    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("Evaluation pipes")
    figure.suptitle("DQN training progress")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    global target_net
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    output_dir = Path(__file__).resolve().parent
    gif_dir = output_dir / "assets" / "gifs"
    plot_dir = output_dir / "assets" / "plots"
    gif_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "deep_q_network_checkpoint.pt"
    metrics_path = output_dir / "training_metrics.json"
    training_plot = plot_dir / "training_progress.png"

    policy_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(
        policy_net.parameters(), lr=CONFIG["learning_rate"]
    )
    replay = ReplayBuffer(CONFIG["replay_capacity"])

    steps_seen = 0
    episode = 0
    training_history = []
    evaluation_history = []
    if checkpoint.exists():
        policy_net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        target_net.load_state_dict(policy_net.state_dict())
        if metrics_path.exists():
            previous_metrics = json.loads(metrics_path.read_text())
            steps_seen = int(previous_metrics.get("training_steps", 0))
            episode = int(previous_metrics.get("episodes", 0))
            training_history = previous_metrics.get("training_history", [])
            evaluation_history = previous_metrics.get("evaluation_history", [])
        print(f"resuming from episode={episode} steps={steps_seen}", flush=True)

    recent_scores = deque(maxlen=50)
    next_evaluation_steps = steps_seen + CONFIG["evaluation_interval"]
    while steps_seen < CONFIG["training_steps"]:
        episode += 1
        epsilon = max(
            CONFIG["min_epsilon"],
            1.0
            - (1.0 - CONFIG["min_epsilon"])
            * steps_seen
            / CONFIG["epsilon_decay_steps"],
        )
        env = FlappyBirdEnv(seed=SEED + episode)
        reward, pipes, steps, losses = run_episode(
            env, policy_net, epsilon, train=True, replay=replay, optimizer=optimizer
        )
        env.close()
        steps_seen += steps
        recent_scores.append(pipes)
        training_history.append(
            {
                "episode": episode,
                "training_steps": steps_seen,
                "reward": float(reward),
                "pipes": int(pipes),
                "episode_length": int(steps),
                "epsilon": float(epsilon),
                "loss": float(np.mean(losses)) if losses else None,
            }
        )
        if steps_seen % CONFIG["target_update_interval"] < steps:
            target_net.load_state_dict(policy_net.state_dict())
        if episode % CONFIG["log_interval_episodes"] == 0:
            print(
                f"episode={episode:5d} steps={steps_seen:7d} epsilon={epsilon:.3f} "
                f"recent_pipes={np.mean(recent_scores):.2f} "
                f"loss={np.mean(losses) if losses else float('nan'):.4f}",
                flush=True,
            )
        if steps_seen >= next_evaluation_steps:
            evaluation_scores = evaluate_policy(
                policy_net, CONFIG["evaluation_seeds"]
            )
            evaluation_median = float(np.median(evaluation_scores))
            evaluation_history.append(
                {
                    "training_steps": steps_seen,
                    "mean_pipes": float(np.mean(evaluation_scores)),
                    "median_pipes": evaluation_median,
                    "scores": evaluation_scores,
                }
            )
            print(
                f"evaluation steps={steps_seen} "
                f"mean_pipes={np.mean(evaluation_scores):.2f} "
                f"median_pipes={evaluation_median:.2f}",
                flush=True,
            )
            if evaluation_median >= CONFIG["target_median_pipes"]:
                print(
                    f"target reached: median_pipes={evaluation_median:.2f} "
                    f">= {CONFIG['target_median_pipes']}",
                    flush=True,
                )
                break
            next_evaluation_steps += CONFIG["evaluation_interval"]

    torch.save(policy_net.state_dict(), checkpoint)

    def random_selector(_):
        return random.randrange(ACTION_COUNT)

    def trained_selector(observation):
        return choose_action(policy_net, observation, epsilon=0.0)

    before_frames, before_reward = record_episode(
        FlappyBirdEnv(seed=CONFIG["comparison_seed"]), random_selector
    )
    after_frames, after_reward = record_episode(
        FlappyBirdEnv(seed=CONFIG["comparison_seed"]), trained_selector
    )
    save_gif(before_frames, gif_dir / "before_training.gif", CONFIG["gif_fps"])
    save_gif(after_frames, gif_dir / "after_training.gif", CONFIG["gif_fps"])

    scores = evaluate_policy(policy_net, CONFIG["evaluation_seeds"])
    if not evaluation_history or evaluation_history[-1]["training_steps"] != steps_seen:
        evaluation_history.append(
            {
                "training_steps": steps_seen,
                "mean_pipes": float(np.mean(scores)),
                "median_pipes": float(np.median(scores)),
                "scores": scores,
            }
        )

    metrics = {
        "config": CONFIG,
        "training_steps": steps_seen,
        "episodes": episode,
        "training_history": training_history,
        "evaluation_history": evaluation_history,
        "evaluation_seeds": CONFIG["evaluation_seeds"],
        "evaluation_pipes_mean": float(np.mean(scores)),
        "evaluation_pipes_median": float(np.median(scores)),
        "evaluation_pipes": scores,
        "before_reward": before_reward,
        "after_reward": after_reward,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_training_plot(training_history, evaluation_history, training_plot)
    print(f"saved checkpoint: {checkpoint}")
    print(f"evaluation mean pipes: {metrics['evaluation_pipes_mean']:.2f}")
    print(f"evaluation median pipes: {metrics['evaluation_pipes_median']:.2f}")
    print(f"baseline reward: {before_reward:.2f}")
    print(f"trained reward: {after_reward:.2f}")


if __name__ == "__main__":
    main()
