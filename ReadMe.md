# ReadMe

## 1. Overview
This repository contains the implementation of the classical **Deep Q-Network** [(DQN)](https://daiwk.github.io/assets/dqn.pdf) algorithm, tested on
[Mountain Car Continuous](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) 
environment from the `gymnasium` package.

Some part of this code was originally part of an introduction to deep reinforcement learning 
presented to Predictive Layer SA. 


It is shared with the hope that it may be useful to someone.


## 2. Organization
This project contains the following python files. 
* `buffer.py:` replay buffer, used to store and sample experiences for training the DQN agent.
* `DQN.py:`  Deep Q-Network (DQN) agent, implemented using the double Q-learning trick.
* `Env:` a simple wrapper used to customize the environment.
* `main.py:` main script.
* `utils.py:` utility functions.

After running the main script, two new folders should appear.

* `logs/:` log files recording training metrics.
* `Videos:` stores videos of episodes performed at different stages of training.



## 3. Usage

### 3.1 Running the code
To train and evaluate a DQN agent, simply execute the `main.py` script. Hyperparameters can be 
customized by using command-line arguments:

```commandline
main.py --hyperparameter_name hyperparameter_value
```
Alternatively, all default values can be directly modified in the code. 

Please, refer to the main.py file for the full 
list of available hyperparameters.

### 3.2 Visualization
Every `--visualize_every` training steps, an episode is run, and its video is saved in the `Videos/` folder.
This provides a visual representation of the agent's progress over time.


During evaluation, the cumulative rewards achieved by the agent are saved in the
`logs/CURRENT_TIME` directory.


### 3.3 Viewing Logs


To view the training logs, ensure you have TensorBoard installed (otherwise `pip install tensorboard`
should work). Then run
```commandline
tensorboard --logdir PATH_TO_LOG_FOLDER
```
Access the dashboard via the URL displayed in your terminal (usually something that http://localhost:6006/).


## 4. Notes and limitation
The default hyperparameter settings in main.py have not been extensively tuned and may not yield a smooth learning curve. 
Feel free to test other values.





