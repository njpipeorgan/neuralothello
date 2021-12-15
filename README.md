# A neural network engine for the game of Othello

This is an archive of the project in which neural networks were built and trained on the game of [Othello (Reversi)](https://en.wikipedia.org/wiki/Reversi).
The engine uses MCTS to explore the positions and play the game based on the evaluations by the networks. 

**Network**: The most recent network has four 32-channel residual blocks. It is trained on 2500 games per epoch with a sampling ratio of 3.

**Self-play**: Games are play between two copies of the newest network. 
The number of evaluations per move is determined by a 
[KLD](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) gain of 0.0004, which corresponds to 2000 evaluations on average.

**Strength**: After over 200 epochs of learning, the best network (200 evals per move) performs ~1750 Elo better than a random player.
Compred to [edax](https://github.com/abulmo/edax-reversi), the engine reaches a similar strength when the node ratio reaches 1:100 
(800 evals by the network ≈ 80000 evals by edax).

![Elo history](/run5/elo-history.png)
