import numpy as np
import gym
class Mountain_Task:
    '''
        Implement the Task interface from the Udacity quadcopter project for
        MoutainCarContinuousV0 env from OpenAI.

        -1 for each time step, until the goal position of 0.45 is reached. As with MountainCarContinuous v0,
        there is no penalty for climbing the left hill, which upon reached acts as a wall.

        Note: The documentation says the environment terminates at position 0.5. Experimentation
        shows that in fact it appears to terminate between 0.45 and 0.5.

        Starting State
        Random position from -0.6 to -0.4 with no velocity.

        Episode Termination
        The episode ends when you reach 0.5 position, or if 200 iterations are reached.

        Solved Requirements
        Average return > 90 over past 100 episodes
    '''
    def __init__(self, *args, **kwargs):
        self.success = False
        self.action_repeat = kwargs['action_repeat']
        self.env = gym.make('MountainCarContinuous-v0')
        self.state_size = self.env.observation_space.shape[0] * self.action_repeat
        self.action_size = self.env.action_space.shape[0]
        self.action_high = self.env.action_space.high[0]
        self.action_low = self.env.action_space.low[0]
        self.monitor = False
        self.i = 0

    def preprocess_state(self, state):
        # mapping the state values to [-1,1]
        return state
        s = np.array(state)
        s[0] = ((state[0] + 1.2) / 1.8) * 2 - 1
        s[1] = ((state[1] + 0.07) / 0.14) * 2 - 1
        return s

    def get_reward(self, state, reward, done):
        """
            Custom reward function for testing purposes.
        """

        # simple reward = position. Don't explicitly incentivise agent to take left hill as well as right hill
        # let it figure it out by itself
        if done and self.success:
            return 100
        else:
            return (state[0]-0.5)/3

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        reward_all = 0
        pose_all = []
        raw_states = []
        for _ in range(self.action_repeat):
            state, reward, done, _ = self.env.step(action) # run up the mountain

            processed_state = self.preprocess_state(state)
            raw_states.append(state)

            if done and self.i < 200:
                self.success = True

            reward_all += reward
            pose_all.append(processed_state)

            self.i += 1

            if done:
                missing = self.action_repeat - len(pose_all)
                pose_all.extend([pose_all[-1]] * missing)
                break

        next_state = np.concatenate(pose_all)
        return next_state, reward_all, done, raw_states

    def reset(self):
        """Reset the env to start a new episode."""
        self.success = False
        self.i = 0
        if self.monitor:
            self.env = gym.wrappers.Monitor(self.env, "./mountaincar-monitor", force=True)
        state = self.env.reset()
        state = self.preprocess_state(state)
        state = np.concatenate([state] * self.action_repeat)
        return state
