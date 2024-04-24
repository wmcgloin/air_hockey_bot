

GAME FILES:
* main.py
* game.py
* paddle.py
* puck.py*

To play game, run `main.py` in terminal. Can optionally pass in --mode "pvp", "pve", or "random" to play against another player, a bot, or watch two bots play 
against each other.

To play against trained model, run `play.py` with your desired model file. Change file path in the script.


RL FILES:
* air_hock_env.py
* test_env.py
* test_env2.py
* dqn.py

Air hockey environment is implemented in `air_hock_env.py`. To test the environment, run `test_env.py` or `test_env2.py`. `dq.py` contains the DQN implementation and trains the model.

Current Reward Structure: 3 points for a win (which is now 2 goals for training), 1 point for a goal, 0.33 points for hitting the puck, -1 point for getting scored on, and -3 points for losing.

Additional Note: the puck and paddle speeds have all been increased by a factor of 5 to expedite training.


