# `train_deep_q_network.py`

This guide explains the Stage 3 CNN-DQN training script using vocabulary from Stages 1, 2, and 3.

The script trains an **agent** to interact with the Flappy Bird **environment**. At time step `t`, the agent receives a visual **state representation**, selects an **action**, receives a **reward**, and observes the next state:

~~~text
s_t → a_t → r_{t+1}, s_{t+1}
~~~

One complete interaction sequence is an **episode**. It ends when the environment returns `terminated=True` or `truncated=True`.

## 1. Stage 1–3 vocabulary in this script

| Curriculum term | Meaning in this script |
|---|---|
| State, action, reward | The visual observation, `0 = do nothing` or `1 = flap`, and feedback from the environment. |
| Transition / episode | `(state, action, reward, next_state, done)`; an episode ends at a terminal or truncated state. |
| Policy | The rule for choosing an action; `ε`-greedy during training and greedy during evaluation. |
| `V(s)` and `Q(s,a)` | Value from a state versus value after taking a particular action; this DQN estimates `Q(s,a)`. |
| Return and discount `γ` | Accumulated reward, with `γ=0.95` weighting future rewards in the update. |
| Bellman / TD update | Move the current Q-value toward the TD target and reduce the TD error. |
| Replay / target networks | Reuse random past transitions and use a delayed network for more stable targets. |
| Evaluation | Measure the greedy policy with `ε=0` across repeated seeds and compare it with a random baseline. |

Stage 1 used a Q-table. Stage 3 replaces that table with a CNN that approximates the Q-function from visual observations.

The Stage 1 distinction still applies: `V(s)` asks how good a state is, while `Q(s,a)` asks how good a particular action is in that state. The CNN outputs Q-values, and the policy chooses the action with the highest one.

## 2. Actions and the visual state

The environment has two actions:

~~~python
ACTION_COUNT = 2              # 0 = do nothing, 1 = flap
OBSERVATION_SHAPE = (4, 42, 42)
~~~

The agent does not receive hidden coordinates such as the bird's position or velocity. It receives four consecutive processed frames so the observation contains information about motion. This is the Stage 2 visual-observation constraint: decisions must be based on pixels rather than privileged state information.

The CNN returns one Q-value for each action:

~~~text
Q(s_t, ·) = [Q(s_t, do nothing), Q(s_t, flap)]
~~~

For example:

~~~text
Q(s_t, ·) = [1.2, 2.0]
greedy action = flap
~~~

The greedy policy chooses `flap` because `2.0 > 1.2`.

## 3. Configuration

The current script uses:

~~~python
SEED = 7
ACTION_COUNT = 2
OBSERVATION_SHAPE = (4, 42, 42)
TRAINING_STEPS = 800_000
EPSILON_DECAY_STEPS = 60_000
MIN_EPSILON = 0.05
EVALUATION_INTERVAL = 10_000
TARGET_MEDIAN_PIPES = 30
EVALUATION_SEEDS = range(200, 220)
~~~

Epsilon decays from about `1.0` to `0.05` during the first `60,000` environment steps. Evaluation runs every `10,000` steps and can stop training early when the median score reaches `30` pipes. The Adam learning rate is `1e-3`; it controls how strongly each TD-error update changes the CNN weights.

## 4. Preprocessing observations

Stage 2 converts each rendered RGB frame into a normalized grayscale frame:

~~~python
def preprocess(frame):
    grayscale = frame.mean(axis=2).astype(np.float32) / 255.0
    return grayscale[::2, ::2]
~~~

An RGB frame has shape `(84, 84, 3)`. Averaging the colour channels and downsampling produces `(42, 42)`. Four frames are stacked into the CNN observation:

~~~text
one processed frame:  (42, 42)
CNN observation:      (4, 42, 42)
~~~

At the start of an episode, the first frame is repeated four times. Afterwards, the history slides:

~~~text
[A, B, C, D] → [B, C, D, E]
~~~

## 5. CNN as a Q-function approximator

`QNetwork` maps a visual observation to two action values. Convolution layers extract visual features; the linear head maps them to estimates of `Q(s,a)`.

~~~python
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
~~~

The shape changes are:

~~~text
(N, 4, 42, 42)
  ↓ convolution
(N, 16, 19, 19)
  ↓ convolution
(N, 32, 9, 9)
  ↓ flatten and linear layers
(N, 2) = [Q(s, do nothing), Q(s, flap)]
~~~

Unlike a Q-table, the CNN learns weights and generalizes across visual observations.

## 6. Essential training loop

The rest of the script is the Stage 1 Q-learning loop adapted to visual observations:

~~~text
observation s_t
→ choose action a_t with ε-greedy policy
→ environment returns reward r_{t+1} and next observation s_{t+1}
→ store transition in replay buffer
→ sample a random mini-batch
→ calculate TD target
→ update the policy CNN
~~~

A transition is stored as:

~~~python
Transition = namedtuple("Transition", "state action reward next_state done")
~~~

Here `done` is true when `terminated or truncated` is true. A finished transition must not use a future Q-value.

The replay buffer breaks up the correlation between consecutive frames and lets the network reuse older transitions.

## 7. The DQN update

For each sampled transition, the policy network estimates the Q-value of the action actually taken. The target network estimates the best next-state Q-value:

~~~python
current_q = policy_net(states).gather(
    1, actions.unsqueeze(1)
).squeeze(1)

with torch.no_grad():
    next_q = target_net(next_states).max(dim=1).values
    targets = rewards + 0.95 * next_q * (1.0 - done)
~~~

This is the Stage 1 Q-learning target:

~~~text
target = r_{t+1} + γ max_a' Q_target(s_{t+1}, a')
~~~

The script uses `γ=0.95`. When `done=1`, the target becomes only the final reward.

The **TD error** is:

~~~text
TD error = target − current Q-value
~~~

The loss measures this error, and backpropagation updates the policy-network weights:

~~~python
loss = F.smooth_l1_loss(current_q, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
~~~

The target network is a delayed copy of the policy network. It is refreshed periodically so the TD target changes more slowly and learning is more stable.

## 8. Training, return, and evaluation

During training, `ε` gradually decreases, shifting behaviour from exploration to exploitation. The script records:

- episode return: the accumulated reward for one episode;
- episode length: how long the bird survives;
- `pipes_passed`: how effectively the agent navigates obstacles;
- loss: the current TD prediction error.

The environment's reward is a learning signal: small alignment or survival rewards make the signal denser, passing a pipe gives positive feedback, and a collision gives negative feedback. The logged episode return is undiscounted:

~~~text
G_t = r_{t+1} + r_{t+2} + r_{t+3} + …
~~~

Evaluation uses the greedy policy with `ε=0` across seeds `200` through `219`. Repeated evaluation scores are stronger evidence than one lucky episode.

After training, the script saves:

~~~text
deep_q_network_checkpoint.pt learned policy-network weights
training_metrics.json        evaluation and before/after metrics
assets/gifs/before_training.gif  random-policy baseline
assets/gifs/after_training.gif   trained greedy-policy episode
~~~

The before-and-after recordings use the same environment seed and settings. The trained agent should achieve higher reward, pass more pipes, and survive longer across repeated evaluation episodes.
