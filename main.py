import argparse
import os
from DQN import DQN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()

    # Random seed
    parser.add_argument('--seed_value', action='store', type=int, default=42)

    """
    Model architecture -> A simple MLP
    """

    parser.add_argument('--num_units', action='store', type=int, default=[24, 24],
                        help='num of units for each layers if --architecture is set to \'MLP\'.')
    parser.add_argument('--dropout_rate', action='store', type=float, default=0.5,
                        help='dropout rate used in each layer of the MLP.')
    parser.add_argument('--output_activation', action='store', type=str, default='linear',
                        help='activation function the last layer.')

    """
    Training hyperparameters
    """
    parser.add_argument('--gamma', action='store', type=float, default=0.95,
                        help='Discount rate.')
    parser.add_argument('--buffer_size', action='store', type=int, default=50000,
                        help='length of the replay buffer.')
    parser.add_argument('--batch_size', action='store', type=int, default=32)
    parser.add_argument('--first_random_steps', action='store', type=int, default=10000,
                        help='number of random steps used to initialize the replay buffer.')
    parser.add_argument('--num_training_steps', action='store', type=int, default=40000)
    parser.add_argument('--learning_rate', action='store', type=float, default=0.001)
    parser.add_argument('--tau', action='store', type=float, default=0.005,
                        help='Tau parameter for the soft update of the target Q-functions')
    parser.add_argument('--rollout_every', action='store', type=int, default=1,
                        help='For each rollout_every training steps, we perform an episode using the current trained'
                             'model and add transitions to the replay buffer.')
    parser.add_argument('--epsilon_value', action='store', type=float, default=1,
                        help='First, epsilon value used for performing the epsilon-greedy policy')
    parser.add_argument('--epsilon_decay', action='store', type=float, default=0.995,
                        help='Value used for epsilon decay at each training steps.')
    parser.add_argument('--min_epsilon', action='store', type=float, default=0.001,
                        help='Min epsilon value used during the rollout.')

    parser.add_argument('--env_step_every', action='store', type=str, default=1,
                        help='for each --env_step_every gradient steps, we perform a rollout '
                             '(i.e. make the agent interact with the environment for one episode and add the data to'
                             'the replay buffer).')
    parser.add_argument('--loss_function', action='store', type=str, default='mse',
                        help='possible values : mse')

    # Evaluations
    parser.add_argument('--evaluate_every', action='store', type=int, default=200,
                        help='evaluate the current trained agent at each --evaluate_every gradient steps.')
    parser.add_argument('--num_episodes_per_evaluation_step', action='store', type=int, default=10,
                        help='Number of episode used to perform one evaluation step. It might me interesting to use'
                             'more than one episode since the first state is randomly initialized.')
    parser.add_argument('--visualize_every', action='store', type=int, default=1000,
                        help='Each --visualize_every, we perform an episode in human mode, and we save the episode into'
                             'a short video.')

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # loading hyperparameters
    args = parse_args()

    # Building DQN agent
    agent = DQN(args=args)

    # Agent training
    agent.train()







