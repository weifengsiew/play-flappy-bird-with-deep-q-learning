# `train_deep_q_network.py`

The script trains an **agent** to interact with the Flappy Bird **environment**. At time step `t`, the agent receives a visual **state representation**, selects an **action**, receives a **reward**, and observes the next state:

~~~text
s_t → a_t → r_{t+1}, s_{t+1}
~~~

One complete interaction sequence is an **episode**. It ends when the environment returns `terminated=True` or `truncated=True`.

## 1. Reinforcement learning vocabulary

| Terms | Relevance to this script |
|---|---|
| State, action, reward | The visual observation, `0 = do nothing` or `1 = flap`, and feedback from the environment. |
| Transition / episode | `(state, action, reward, next_state, done)`; an episode ends at a terminal or truncated state. |
| Policy | The rule for choosing an action; `ε`-greedy during training and greedy during evaluation. |
| `V(s)` and `Q(s,a)` | Value from a state versus value after taking a particular action; this DQN estimates `Q(s,a)`. |
| Return and discount `γ` | Accumulated reward, with `γ=0.95` weighting future rewards in the update. |
| Bellman / TD update | Move the current Q-value toward the TD target and reduce the TD error. |
| Replay / target networks | Reuse random past transitions and use a delayed network for more stable targets. |
| Evaluation | Measure the greedy policy with `ε=0` across repeated seeds and compare it with a random baseline. |

## 2. Actions and state

Two actions:
~~~python
ACTION_COUNT = 2              # 0 = do nothing, 1 = flap
~~~

State:
~~~python
OBSERVATION_SHAPE = (4, 42, 42)
~~~

State does not include hidden coordinates such as the bird's position or velocity. It receives four consecutive processed frames so the observation contains information about motion.

## 3. Choosing best action and state using Q-values

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

## 4. Configuration

Training parameters are stored in [`config.json`](config.json), so experiments can change settings without editing the training code. The most important parameters are:

| Parameter | Default | Purpose |
|---|---:|---|
| `training_steps` | `800000` | Maximum number of environment steps. |
| `learning_rate` | `0.001` | Size of each Adam optimizer update. |
| `discount_factor` | `0.95` | Weight given to future rewards. |
| `batch_size` | `64` | Number of replay transitions per update. |
| `replay_capacity` | `50000` | Maximum number of stored transitions. |
| `epsilon_decay_steps` | `60000` | Steps over which exploration decreases. |
| `min_epsilon` | `0.05` | Minimum probability of a random action. |
| `evaluation_interval` | `10000` | Steps between greedy evaluations. |
| `target_median_pipes` | `30` | Early-stopping evaluation target. |

Epsilon decays from about `1.0` to `0.05` during the first `60,000` environment steps. Evaluation runs every `10,000` steps and can stop training early when the median score reaches `30` pipes.

## 5. Tracking performance across episodes

After each episode, the script records the episode number, environment steps, reward, pipes passed, episode length, epsilon, and mean training loss. At each evaluation checkpoint it records the mean and median number of pipes across the fixed evaluation seeds.

These records are saved in `training_metrics.json`:

- `training_history` contains noisy per-episode training results.
- `evaluation_history` contains greedy-policy results at regular environment-step checkpoints.

The script also writes `assets/plots/training_progress.png`, which shows:

1. Episode reward and its rolling mean.
2. Pipes passed and its rolling mean.
3. Evaluation mean and median pipes against environment steps.

The rolling mean uses the `rolling_window` value from `config.json` and makes the overall learning trend easier to see than individual episodes alone.

## 6. Preprocessing observations

Each rendered RGB frame is preprocessed into a normalized grayscale frame:

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

## 7. CNN as a Q-function approximator

The CNN's job is to provide a differentiable function with parameters `θ` that maps a state to one Q-value per action:

~~~text
Q_θ(s, ·) = [Q_θ(s, do nothing), Q_θ(s, flap)]
~~~

The parameters `θ` include all convolution and linear-layer weights. Training changes `θ` so that these outputs become useful estimates of Q-values for each action. The network is not trained with a label saying which action is correct. Instead, the label is constructed from the reward and the estimated value of what happens next.

~~~python
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Extract spatial features from the four stacked grayscale frames.
        self.features = nn.Sequential(
            # Four input frames become 16 learned feature maps.
            nn.Conv2d(4, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            # Combine low-level features into 32 higher-level feature maps.
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        # Convert visual features into one Q-value for each action.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_COUNT),
        )

    def forward(self, observations):
        # Return Q(s, a) for every available action.
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

## 8. Essential training loop

The rest of the script is the Q-learning loop adapted to visual states:

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

Here `done` is true when `terminated or truncated` is true. A finished transition must not use a future Q-value. The replay buffer breaks up the correlation between consecutive frames and lets the network reuse older transitions.

## 9. How one experience updates the action values

This section shows how replayed experiences train the CNN: compute a target, compare it with the current Q-value, and update the policy network. Repeating this improves action selection.

`QNetwork` is the CNN definition. The code creates two instances:

~~~python
policy_net = QNetwork()
target_net = QNetwork()
~~~

- **Policy network (`policy_net`):** the trainable CNN used to choose actions from the current state; its weights are updated later using sampled past experiences from replay.
- **Target network (`target_net`):** a delayed copy used only to calculate training targets.

The timing is:

~~~text
current state → policy chooses action → store transition
                later: replay samples transitions → update policy network
~~~

### Q-network: predict action values

The CNN defined by `QNetwork` predicts the expected discounted future reward for each possible action.

### Policy network: choose and learn

`policy_net` is the trainable CNN. Its predictions help choose actions and provide the current Q-value for the loss. For a replay transition, the loss selects only the value for the action taken:

~~~python
current_q = policy_net(states).gather(
    1, actions.unsqueeze(1)
).squeeze(1)
~~~

If the sampled action is `flap`, `gather` selects `Q_θ(s_t, flap)`.

### Target network: provide a stable target

`target_net` is a delayed copy of the policy network. Its next-state predictions provide the target; it is not updated by backpropagation:

~~~python
with torch.no_grad():
    next_q = target_net(next_states).max(dim=1).values
targets = rewards + 0.95 * next_q * (1.0 - done)
~~~

For a non-terminal transition, the target is:

~~~text
y_t = r_{t+1} + γ max_{a'} Q_target(s_{t+1}, a')
~~~

If `done=1`, the target is only the final reward:

~~~text
y_t = r_{t+1}
~~~

The `max` forms the target; `argmax` later chooses the action with the highest estimated long-term return:

~~~text
argmax_a Q_θ(s, a)
~~~

This may not be the action with the highest immediate reward.
