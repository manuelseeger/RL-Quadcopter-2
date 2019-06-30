import numpy as np

class Random_Binary_Agent():
    '''
        Random Binary Agent

        Agent chooses either the minimum or maximum rotor speed at random, per rotor, and does not change
        this policy during an episode.
        If the episode yield higher total rewards than previous highest reward, agent adopts the last policy
        as a candidate policy. It will repeat this policy n times (configurable), before discarding it and
        returning to it's exploration mode.

        This was supposed to be a joke agent to benchmark the DDPG agent against in the take-off task.
        It remains unbeaten.
    '''
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.best_score = -np.inf
        self.exploration_policy = np.zeros(self.action_size) + self.action_low

        self.policy = None
        self.count = 0
        self.current_best_score = -np.inf
        self.best_score = -np.inf

        # Episode variables
        self.reset_episode()

    def configure(self, policy_attempts):
        self.policy_attempts = policy_attempts

    def reset_episode(self):
        self.total_reward = 0.0
        self.exploration_policy = np.random.randint(2, size=self.action_size) * self.action_high + self.action_low
        self.exploration_policy = np.clip(self.exploration_policy, self.action_low, self.action_high)
        state = self.task.reset()
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.total_reward += reward

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state, p=0):
        if self.policy is not None:
            return self.policy
        else:
            return self.exploration_policy

    def learn(self):
        # Learn by comparing to highest reward:
        if self.total_reward > self.current_best_score:
            if self.total_reward > self.best_score:
                self.best_score = self.total_reward
                self.best_policy = self.exploration_policy

            self.current_best_score = self.total_reward
            self.policy = self.exploration_policy

            self.count = 0
            print('new best score {}, new policy {} {} {} {}'.format(self.best_score, *self.policy))
        else:
            if self.count > self.policy_attempts:
                self.current_best_score = -np.inf
                self.policy = None
                self.count = 0
            else:
                self.count += 1

