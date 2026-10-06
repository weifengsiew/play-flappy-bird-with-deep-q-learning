# Reinforcement Learning — Learning Notes

## Concepts and checkpoints

### Reinforcement learning

- An agent learns to make decisions by interacting with an environment.
- The loop is: agent takes an action → receives reward as feedback → learns which actions lead to better results.
- Example: winning a chess game can be positively rewarded; losing can be negatively rewarded.

**Question:** What is reinforcement learning?

**Correct answer:** It is learning to make decisions through interaction, feedback, and improving future actions.

### State-transition notation

- `S_t`: current state at time `t`, such as the current chessboard position.
- `A_t`: action at time `t`, such as a chess move.
- `R_{t+1}`: reward received at time `t+1`, such as feedback for winning or losing.
- `S_{t+1}`: new state at time `t+1`, such as the new chessboard position.
- A trajectory follows `S_t → A_t → R_{t+1}, S_{t+1}`.

**Question:** What do `S_t`, `A_t`, `R_{t+1}`, and `S_{t+1}` represent?

**Correct answer:** Current state, action taken, reward received, and new state, respectively.

### Reward, return, and discounting

- A reward is feedback from one step.
- The return is accumulated future reward from the current time onward.
- A discount factor `γ` controls how much future rewards matter.
- A reward received two steps from now is weighted by `γ²`.

Numerical example:

```text
γ = 0.5
future reward = 20
discounted value = 0.5² × 20 = 5
```

**Question:** If rewards are `0, 0, 0, +1`, what is the undiscounted return immediately before the final move?

**Correct answer:** `1`, because the return includes the upcoming `+1` reward.

### Policies

- A policy is the agent's strategy for choosing actions.
- `π(a|s)` means the probability of choosing action `a` given state `s`.
- A deterministic policy always chooses the same action in a state.
- A stochastic policy assigns probabilities to multiple actions.

**Question:** A policy chooses action A 80% of the time and action B 20% of the time. Is it deterministic or stochastic?

**Correct answer:** Stochastic.

### Value functions

- `V(s)` estimates expected return starting from state `s` under a policy.
- `Q(s,a)` estimates expected return after taking action `a` in state `s` and then following a policy.
- A greedy policy chooses the action with the highest Q-value.

**Question:** What is the difference between `V(s)` and `Q(s,a)`?

**Correct answer:** `V(s)` estimates return from a state. `Q(s,a)` estimates return from taking a specific action in a specific state.

### Exploration and exploitation

- Exploration tries less-known actions to gather information.
- Exploitation chooses the action currently believed to be best.
- An ε-greedy policy explores randomly with probability `ε` and exploits with probability `1−ε`.
- During training, `ε` is often high at first and gradually reduced.
- During evaluation, `ε=0` is commonly used to measure the greedy policy without random actions.

**Question:** Why does an agent need both exploration and exploitation?

**Correct answer:** It explores when Q-values are uncertain and need updating; it exploits by choosing the best action currently known.

### Episodes and terminal states

- An episode is one complete interaction sequence.
- A terminal state ends the episode, including a win, loss, draw, or checkmate.
- There is no future value after a terminal state.

**Question:** When is a chess position terminal?

**Correct answer:** When the game ends, including a win, loss, draw, or checkmate. An ordinary ongoing position is non-terminal.

### Markov decision processes

- An MDP is described by `(S, A, P, R, γ)`:
  - `S`: possible states.
  - `A`: possible actions.
  - `P(s'|s,a)`: probability of reaching next state `s'` given state `s` and action `a`.
  - `R`: reward rule.
  - `γ`: discount factor.
- The current state should contain enough information to choose an action and predict what may happen next.

**Question:** In a chess MDP, what do `S`, `A`, and `P(s'|s,a)` represent?

**Correct answer:** `S` is the set of possible board positions, `A` is the set of possible moves, and `P(s'|s,a)` is the probability of reaching `s'` given `s` and `a`.

### Bellman idea and expected value

- A state's value can be understood as immediate reward plus discounted future value: `V(s) ≈ r + γV(s')`.
- When outcomes are uncertain, expected value weights each possible outcome by its probability.

Numerical example:

```text
immediate reward = 2
next-state value = 10
γ = 0.9
estimated value = 2 + 0.9 × 10 = 11
```

**Question:** Why do we use expected value when the next state is uncertain?

**Correct answer:** Each possible outcome is weighted by its probability, so more likely states contribute more to the estimate.

### Q-learning

- For a non-terminal transition, the target is `r + γ max Q(s',a')`.
- The TD error is `target − current Q-value`.
- The update is `Q_new = Q_old + α × TD error`.
- `γ` controls future-reward importance.
- `α` is the learning rate and controls how quickly the estimate changes.
- For a terminal transition, the target is just `r`; no future Q-value is included.
- The update process is: calculate target → calculate TD error → calculate correction `α × TD error` → update the Q-value.

Numerical example:

```text
old Q-value = 4
target = 10
α = 0.5
TD error = 10 − 4 = 6
new Q-value = 4 + 0.5 × 6 = 7
```

**Question:** Should a terminal winning transition include a future Q-value in its target?

**Correct answer:** No. There is no future state after the episode ends, so the target is just the immediate reward.

### Hallway environment

- Positions are `0, 1, 2, 3, 4`; the agent starts at `0` and the goal is `4`.
- Actions are `left` and `right`.
- Hitting the boundary leaves the agent in the same position.
- Moving from `3` to `4` returns `(4, +1, True)`.
- Moving from `2` to `3` returns `(3, 0, False)`.
- `reset()` returns `(0, 0, False)` in our simple interface.
- An episode stops when the agent reaches position `4`.

**Question:** What does moving right from position `3` return?

**Correct answer:** `(4, +1, True)`: next state `4`, reward `+1`, and terminal status `True`.

### Q-table matrix

- A Q-table has one row per state and one column per action.
- With five states and two actions, it has shape `(5, 2)` and contains `10` Q-values.

Numerical example:

```text
Q = [[0.0, 0.0],
     [0.2, 0.5],
     [0.4, 0.7],
     [0.6, 1.0],
     [0.0, 0.0]]
```

Row `2` is `[0.4, 0.7]`, so `Q(2,left)=0.4` and `Q(2,right)=0.7`; a greedy agent chooses right. Initial Q-values of `0` mean there is no learned preference yet.

**Question:** How many Q-values are in this table, and which action does a greedy agent choose in state `2`?

**Correct answer:** There are `5 × 2 = 10` Q-values, and the greedy agent chooses right because `0.7 > 0.4`.

### State vector

A one-hot state vector for state `2` in a five-state hallway is:

```text
s = [0, 0, 1, 0, 0]
```

The `1` marks the current state and the `0`s mark all other states. We are using the Q-table directly for now; this vector is a numerical example of state-vector notation.

**Question:** What does the `1` represent in the vector `[0, 0, 1, 0, 0]`?

**Correct answer:** It marks that the agent is currently in state `2`.

### Evaluation

- During evaluation, `ε=0` removes random exploration so the learned greedy policy can be measured.
- Learning updates are typically disabled during evaluation.
- A useful metric is the number of moves needed to reach position `4`.

**Question:** Why set `ε=0` during evaluation, and name one useful metric?

**Correct answer:** `ε=0` removes random exploratory actions; a useful metric is the number of moves taken to reach position `4`.
