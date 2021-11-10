# real-world-reinforcement-learning

The document main.pdf outlines primary considerations for designing production-worthy reinforcement learning algorithms in customer-facing settings. We will (1) establish the notation, (2) discuss how policies are evaluated, (3) show how policies can be defined by evaluation functions alone, (4) demonstrate how we train an evaluation for our policy, even from customer interactions driven by another policy, known as off-policy training, (5) show how we can train the policy directly from customer interactions, and (6) demonstrate how we can combine both approaches for the current state-of-the-art in the field. We also (7) provide practical considerations under the hood for building our policies and evaluation functions, as well as (8) additional resources to help the reader dive more deeply into the field, as well as discover implementations that can be used right now. This document attempts to be brief but mathematically thorough.

The associated code relies on a local version of our Bandito and BanditoSequence APIs. This code was made for research purposes, and in the future we will update the code to cooperate with our online APIs, since we currently do not realease our local APIs.

To run, download and execute in the same directory:

```python
from src.gamend import EvaluateGamePerformance

evaluator_with_q_learning = EvaluateGamePerformance()
evaluator_with_q_learning.run_training()

evaluator = EvaluateGamePerformance(q_learning=False)
evaluator.run_training()
```
