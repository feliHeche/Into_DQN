import gymnasium as gym


class OurEnv:
    """
    Just a simple wrapper used to customize the environment.

    This is just the 'MountainCar-V0' environment with a simplified reward function.
        -> Simplify the training. -> I didnt want to spend too much time for hyperparameter tuning.
    """
    def __init__(self):
        # Basic env -> render mode 'rgb_array' is used to save gif during the training.
        self.env = gym.make('MountainCar-v0', render_mode='rgb_array')

        # Just to avoid using self.env.env into DQN class
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.spec = self.env.spec

    def render(self):
        # return the current frame of the environment
        return self.env.render()

    def reset(self):
        # reset the episode
        return self.env.reset()

    def step(self, action):
        """
        Used to perform one environment step.

        :param action: an allowed action from the environment.
        :return: a tuple (next state, reward, done, info, test) that provides all information related to next state.
        """
        next_state, reward, done, info, test = self.env.step(action=action)

        # recalculate the reward to simplify the training
        reward = (6*next_state[0] + 10*abs(next_state[1]))
        if done:
            reward += 10000

        return next_state, reward, done, info, test

