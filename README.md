The agent operates by maintaining a history of:

Its own actions

The opponent's actions

Encoded sequences of prior moves (2-move and 3-move patterns)

Targets used for model training and evaluation

The strategies include:

 -- 2-sequence predictor: Predicts the next opponent move based on the last 2.

 -- 3-sequence predictor: Same as above, using the last 3 moves.

 -- Counter-strategies: Predicts the opponent's response to the agent’s likely move.

 -- Random strategy: Adds randomness to avoid predictability.

Strategy Selection

 -- Each strategy has an associated weight, which changes dynamically:

 -- Winning strategies gain more weight.

 -- Poor-performing strategies lose weight.

 -- The agent probabilistically selects among the strategies based on these weights.

To run the simulation, ensure you have installed all dependencies:

```bash
pip install -r requirements.txt

python analysis.py
