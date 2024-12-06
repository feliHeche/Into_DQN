import numpy as np


class ReplayBuffer:
    """
    Class used for saving agent-environment interactions of the form (states, actions, rewards, next_states).
    These tuples are then used for training.
    """
    def __init__(self,
                 size: int,
                 obs_dim: int,
                 action_dim: int,
                 seed: int = 0):
        """
        Construct a ReplayBuffer.

        :param size: size of the replay buffer, i.e. maximum length of transitions stored.
        :param obs_dim: dimension of the environment state.
        :param action_dim: dimension of the action.
        :param seed: random seed.
        """

        # the buffer is actually just a dict
        self.buffer = dict()
        self.size = size

        self.buffer['states'] = np.full((self.size, obs_dim), float('nan'), dtype=np.float32)
        self.buffer['actions'] = np.full((self.size, action_dim), float('nan'), dtype=np.float32)
        self.buffer['rewards'] = np.full((self.size, 1), float('nan'), dtype=np.float32)
        self.buffer['next states'] = np.full((self.size, obs_dim), float('nan'), dtype=np.float32)
        self.buffer['terminals'] = np.full((self.size, 1), float('nan'), dtype=np.float32)

        self._stored_steps = 0
        self._write_location = 0
        self._random = np.random.RandomState(seed)

    @property
    def obs_dim(self):
        # dimension of the environment state
        return self.buffer['states'].shape[-1]

    @property
    def action_dim(self):
        # action dimension
        return self.buffer['actions'].shape[-1]

    def __len__(self):
        # number of sample stored in the replay buffer
        return self._stored_steps

    def add_samples(self, states, actions, next_states, rewards, terminals):
        """
        Add some transition into the replay buffer.

        :param states: current states s_t of the environment
        :param actions: actions a_t chosen by the agent
        :param next_states: next environment state s_t+1
        :param rewards: current reward r_t
        :param terminals: boolean which asserts if next_states is the last sate of the episode.
        """

        for obsi, actsi, nobsi, rewi, termi in zip(states, actions, next_states, rewards, terminals):
            self.buffer['states'][self._write_location] = obsi
            self.buffer['actions'][self._write_location] = actsi
            self.buffer['next states'][self._write_location] = nobsi
            self.buffer['rewards'][self._write_location] = rewi
            self.buffer['terminals'][self._write_location] = termi

            self._write_location = (self._write_location + 1) % self.size
            self._stored_steps = min(self._stored_steps + 1, self.size)

    def sample(self, batch_size):
        """
        Sample batch_size transitions from the replay buffer.

        :param batch_size: number of samples.
        :return: a dict containing the sampling.
        The elements stored in this dict ore of dimension [batch_size, element_dim]
        """
        idxs = self._random.choice(self._stored_steps, batch_size)
        obs = self.buffer['states'][idxs]
        actions = self.buffer['actions'][idxs]
        next_obs = self.buffer['next states'][idxs]
        rewards = self.buffer['rewards'][idxs]
        terminals = self.buffer['terminals'][idxs]

        data = {
            'obs': obs,
            'actions': actions,
            'next_obs': next_obs,
            'rewards': rewards,
            'terminals': terminals
        }

        return data

