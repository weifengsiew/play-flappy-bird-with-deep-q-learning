# Reinforcement Learning — Quiz Answers

1. It is learning to make decisions through interaction, feedback, and improving future actions.
2. Current state, action taken, reward received, and new state, respectively.
3. `1`, because the return includes the upcoming `+1` reward.
4. Stochastic.
5. `V(s)` estimates return from a state. `Q(s,a)` estimates return from taking a specific action in a specific state.
6. It explores when Q-values are uncertain and need updating, while exploiting chooses the best action currently known.
7. When the game ends, including a win, loss, draw, or checkmate. An ordinary ongoing position is non-terminal.
8. `S` is the set of possible board positions, `A` is the set of possible moves, and `P(s'|s,a)` is the probability of reaching `s'` given `s` and `a`.
9. Each possible outcome is weighted by its probability, so more likely states contribute more to the estimate.
10. No. There is no future state after the episode ends, so the target is just the immediate reward.
11. `(4, +1, True)`: next state `4`, reward `+1`, and terminal status `True`.
12. There are `5 × 2 = 10` Q-values, and the greedy agent chooses right because `0.7 > 0.4`.
13. It marks that the agent is currently in state `2`.
14. `ε=0` removes random exploratory actions; a useful metric is the number of moves taken to reach position `4`.
