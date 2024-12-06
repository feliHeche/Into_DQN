from buffer import ReplayBuffer
from utils import *
import numpy as np
import random
import datetime
from tqdm import tqdm
import os
import imageio
from Env import OurEnv


class DQN:
    """
    Implementation of a Deep Q-network (DQN) agent presented in "Human-level control through deep
    reinforcement learning".
        See : https://daiwk.github.io/assets/dqn.pdf

    Note that we are using the double Q-learning trick as introduced in "Deep reinforcement learning with double
    Q-learning".
        See : https://arxiv.org/abs/1509.06461
    """

    def __init__(self, args):
        """
        Building a DQN agent.

        :param args: parser containing all hyperparameters. See main.py for more details.
        """

        # save all hyperparameters
        self.args = args

        # environment
        self.env = OurEnv()

        # Building replay buffer B used to store agent-environment interactions
        self.buffer = ReplayBuffer(size=int(args.buffer_size),
                                   obs_dim=self.env.observation_space.shape[0],
                                   action_dim=1)

        # Build Q-models
        self.q = create_mlp(num_units=args.num_units,
                            output_activation=args.output_activation,
                            dropout_rate=args.dropout_rate,
                            output_dim=self.env.action_space.n,
                            name='q')

        self.target_q = create_mlp(num_units=args.num_units,
                                   output_activation=args.output_activation,
                                   dropout_rate=args.dropout_rate,
                                   output_dim=self.env.action_space.n,
                                   name='target_q')

        # copy the weight to the target q-functions : target_theta <- theta
        self._copy_q_weight_to_target()

        # optimizer
        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)

        # loss function
        if self.args.loss_function == 'mse':
            self.loss_function = tf.keras.losses.MeanSquaredError()

        # epsilon value used to perform epsilon-greedy policy
        self.epsilon = self.args.epsilon_value

        # count the number of training step
        self.training_step = 0

        # building logfiles
        self._build_logfiles()

    def train(self):
        """
        Main training loop.
        """
        print('We begin the training of a DQN agent')
        print('Buffer initialisation...')
        # Add a given number of transitions (s_{t}, a_{t}, r_{t}, s_{t+1}) to B. Actions are randomly chosen.
        self.buffer_initialisation()
        print('... DONE')

        bar = tqdm(range(int(self.args.num_training_steps)))
        for _ in bar:
            bar.set_description('DQN training')
            # One step model training
            self.train_one_step()

            # Add one episode of transactions (s_{t}, a_{t}, r_{t}, s_{t+1}) into the replay buffer
            if self.training_step % self.args.rollout_every == 0:
                self.add_one_episode()

            # Model evaluation
            if self.training_step % self.args.evaluate_every == 0:
                self.evaluate()

            # Save one an episode where the actions are chosen by the current trained policy in a gif files.
            if self.training_step % self.args.visualize_every == 0:
                self.visualize_one_episode()

    def buffer_initialisation(self):
        """
        Add args.first_random_steps (states, actions, rewards, next_actions) couples to the buffer self.buffer.
        Actions are randomly chosen.
        """
        steps = 0
        while steps < self.args.first_random_steps:
            done = False
            state, _ = self.env.reset()
            while not done:
                # random sampling
                action = self.env.action_space.sample()
                next_state, rewards, done, _, _ = self.env.step(action=action)
                self.buffer.add_samples(states=[state], actions=[action],
                                        next_states=[next_state], rewards=[rewards], terminals=[done])
                steps += 1
                state = next_state

    def train_one_step(self):
        """
        One training step.
        Update the Q-network, perform the soft update on the target network and perform epsilon decay.
        """

        data = self.buffer.sample(batch_size=self.args.batch_size)

        """
        Compute target Q-values
        """
        # select action according to the q-model (and not the target model!) according to a greedy policy
        next_actions = self.select_action(data['next_obs'], evaluation=True)
        # select corresponding q-values
        next_all_q_values = self.target_q(data['next_obs'])
        next_q_values = tf.gather(next_all_q_values, next_actions, axis=1, batch_dims=1)
        # compute target
        target = data['rewards'] + self.args.gamma * tf.stop_gradient(tf.expand_dims(next_q_values, axis=-1))

        """
        Q-model update
        """
        # compute loss
        with tf.GradientTape() as q_tape:
            all_q_values = self.q(data['obs'])
            q_values = tf.gather(all_q_values, np.asarray(data['actions'], dtype=int), axis=1, batch_dims=1)
            td_error_q = self.loss_function(q_values, target)
        # compute gradient
        grads = q_tape.gradient(td_error_q, self.q.trainable_variables)
        # perform gradient descent
        self.q_optimizer.apply_gradients(zip(grads, self.q.trainable_variables))

        """
        Soft update and epsilon decay
        """

        # target soft update
        soft_update_from_to(source_model=self.q, target_model=self.target_q, tau=self.args.tau)

        # perform epsilon decay
        self.epsilon = max(self.epsilon*self.args.epsilon_decay, self.args.min_epsilon)

        self.training_step += 1

    def _copy_q_weight_to_target(self):
        """
        Small function used to initialize the weight of the target neural network.
        """
        state, _ = self.env.reset()

        # Models building
        state_in_good_shape = tf.expand_dims(state, axis=0)
        self.q(state_in_good_shape)
        self.target_q(state_in_good_shape)

        # copy weight from Q models to target
        self.target_q.set_weights(self.q.get_weights())

    def _build_logfiles(self):
        """
        Create Tensorboard logfiles and save all hyperparmeters into the logfile.
        The title of the logfile it the time when the script has been executed.

        This function also check if there is an folder called ./Videos/ + current_time. If a such
        folder does not exists, we build one. -> We save videos in this folder.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # tensorboard writer
        log_dir = 'logs/' + current_time
        self.summary_writer = tf.summary.create_file_writer(log_dir)

        # saving hyperparameter into logfile
        hyperparameters = [tf.convert_to_tensor([k, str(getattr(self.args, k))]) for k in vars(self.args)]
        with self.summary_writer.as_default():
            tf.summary.text('hyperparameters', tf.stack(hyperparameters), step=0)

        # videos folder
        video_dir = './Videos'
        if not os.path.exists(video_dir):
            # If it doesn't exist, create it
            os.makedirs(video_dir)

        video_dir = video_dir + '/' + current_time
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        self.videos_dir = video_dir

    def evaluate(self):
        """
        Evaluate current trained model and save performances into the logfiles.
        """
        # Since the initial state is randomly assigned we need to perform multiple episodes
        all_episodes = [self._evaluate_one_episode() for _ in range(self.args.num_episodes_per_evaluation_step)]

        # write the mean of these cumulative rewards mean into our logfile
        with self.summary_writer.as_default():
            tf.summary.scalar('Cumulative rewards per episode',
                              np.sum(all_episodes)/len(all_episodes),
                              step=self.training_step)

    def _evaluate_one_episode(self):
        """
        Perform one episode and return the score of this episode.
        Actions are chosen following the greedy policy
            -> action corresponds to the action which maximizes our Q-function in each timestep.

        :return: Score of our agent for one episode.
        """
        cumulative_rewards = 0
        episode_step = 0
        done = False
        state, _ = self.env.reset()
        while not done and episode_step < self.env.spec.max_episode_steps:
            action = self.select_action(state=state, evaluation=True)
            next_state, rewards, done, _, _ = self.env.step(action=action.numpy()[0])
            cumulative_rewards += rewards
            state = next_state
            episode_step += 1

        return cumulative_rewards

    def select_action(self, state, evaluation=False):
        """
        :param state: State provided by the environment expected to be of dimension
                        [batch x 2] or [2].
        :param evaluation: If set to False action action are chosen following an epsilon-greedy approach.
                        If set to True, only a greedy approach. Default value is False.

        :return: The action generated using the policy define with the 'evaluation' parameter.
                Action are returned in a tensor of shape [batch x 1]. If the input is of dim [2], the output dim
                is [1x1].
        """
        if not evaluation and random.uniform(0, 1) < self.epsilon:
            action = tf.expand_dims(tf.convert_to_tensor(self.env.action_space.sample()), axis=0)
        else:
            if len(state.shape) != 2:
                state = tf.expand_dims(state, axis=0)
            q_values = self.q(state)
            action = tf.argmax(q_values, axis=-1)
        return action

    def add_one_episode(self):
        """
        Perform one episode using the epsilon-greedy policy and save transitions of the form
        (s_{t}, a_{t}, r_{t}, s_{t+1}) into the replay buffer.
        """
        done = False
        nb_state = 0
        state, _ = self.env.reset()
        while not done and nb_state < self.env.spec.max_episode_steps:
            action = self.select_action(state=state)
            next_state, rewards, done, _, _ = self.env.step(action=action.numpy()[0])
            self.buffer.add_samples(states=[state], actions=[action],
                                    next_states=[next_state], rewards=[rewards], terminals=[done])
            state = next_state
            nb_state += 1

    def visualize_one_episode(self):
        """
        Perform one episode where the actions are chosen following the current trained policy (not the epsilon-greedy)!
        This episode is then saved into a gif with title corresponding to the number of training steps.
        """
        state, _ = self.env.reset()
        nb_step = 0
        frames = []
        done = False
        while not done and nb_step < self.env.spec.max_episode_steps:
            action = self.select_action(state=state)
            next_state, rewards, done, _, _ = self.env.step(action=action.numpy()[0])
            frame = self.env.render()
            frames.append(label_with_episode_number(frame, episode_num=nb_step))
            state = next_state
            nb_step += 1
        gif_title = str(self.training_step) + '.gif'
        imageio.mimwrite(os.path.join(self.videos_dir, gif_title), frames, fps=60)





