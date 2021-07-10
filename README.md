# real-world-reinforcement-learning

This repository relies on a local version of our Bandito and BanditoSequence APIs. This code was made for research purposes, and in the future we will update the code to cooperate with our online APIs, since we currently do not realease our local APIs.

To run, download and execute in the same directory:

```
python
from src.gamend import EvaluateGamePerformance

evaluator_with_q_learning = EvaluateGamePerformance()
evaluator_with_q_learning.run_training()

evaluator = EvaluateGamePerformance(q_learning=False)
evaluator.run_training()

```
