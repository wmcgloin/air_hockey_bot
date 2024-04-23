

GAME FILES:
* main.py
* game.py
* paddle.py
* puck.py*

To play game, run `main.py` in terminal. Can optionally pass in --mode "pvp", "pve", or "random" to play against another player, a bot, or watch two bots play against each other.

RL FILES:
* air_hock_env.py
* test_env.py
* test_env2.py
* dqn.py

Air hockey environment is implemented in `air_hock_env.py`. To test the environment, run `test_env.py` or `test_env2.py`. `dq.py` contains the DQN implementation and trains the model.

Additional Note: the puck and paddle speeds have all been increased by a factor of 5 to expedite training.



REWARDS:

Goal scored: +1
Goal against: -1
Puck Collision: + 0.33
Win Game: +3
Lose Game: +3