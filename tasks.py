from task import Task


import numpy as np
from physics_sim import PhysicsSim


class TakeOff_Task(Task):
    '''
        Take Off Task

        Initialize agent at 0,0,10 (unless configured differently)
        Position is vertically above initial position, distance configured below.

        Success and failure conditions:
            - Succeeds if agents gets within n units euclidean distance (configurable)
            - Fails if crashes (Z <= 0)
            - Fails if runs out of bounds (euclidean distance to target > 2x initial distance
            - Fails if time runs out

        On success or failure, episode is terminated
    '''
    def __init__(self, *args, **kwargs):
        super(TakeOff_Task, self).__init__(*args, **kwargs)

        self.success = False

        # Goal
        self.target_pos = kwargs['target_pos'] if kwargs['target_pos'] is not None else np.array([0., 0., 10.])
        self.distance_to_target = 0.
        self.init_distance = np.linalg.norm(self.target_pos - self.sim.init_pose[:3])
        self.success_distance = 1

        self.outcome = 0

    def configure(self, action_repeat, action_low, action_high, action_size, target_pos,
                  init_velocities, init_angle_velocities, init_pose, success_distance):
        self.action_low = action_low
        self.action_high = action_high
        self.action_repeat = action_repeat
        self.action_size = action_size
        self.target_pos = target_pos
        self.success_distance = success_distance
        self.state_size = self.action_repeat * 6

    def get_z_reward(self):
        ''''
            Alternative reward function based only on Z-position.
        '''
        reward = 1 - (abs(self.target_pos[2] - self.sim.init_pose[2]) / (self.init_distance * 2)) ** .4
        return reward


    def get_reward(self, done):
        ''''
            General considerations:
                - Keep reward positive at long as the agent is flying to not introduce incentive to suicide
                - Penalize a crash with a large negative
                - Penalize running out of bounds with a large negative
                - Reward reaching the goal with a large positive
                - Reward getting closer to the goal in small steps

            Reward function shaped to be between 0 and 1 as long as the agent is in the air
        '''

        # euclidean distance to target
        self.distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])

        # function modelled after distance as a fraction of out-of-bounds distance, shaped
        # by squaring
        reward = 1 - (self.distance_to_target / (self.init_distance * 2)) ** .5

        #reward = self.get_z_reward()

        reward = reward / self.action_repeat

        # Termination conditions and reward modifiers. These are kept very large to prevent reward hacking
        # like hovering the agent near the target without reaching it fully.
        if done and self.goal_reached():
            reward += 100
        elif done and self.sim.pose[2] <= 0.:
            self.outcome = 2
            reward -= 50
        elif done and self.out_of_bounds():
            self.outcome = 4
            reward -= 30
        elif done and not self.goal_reached() and self.sim.runtime > self.sim.time:
            self.outcome = 1
            reward -= 20
        else:
            self.outcome = 3

        return reward

    def goal_reached(self):
        return self.target_pos[2] - self.sim.pose[2] < self.success_distance
        #return self.distance_to_target < self.success_distance

    def out_of_bounds(self):
        return self.distance_to_target > self.init_distance * 2

    def step(self, rotor_speeds):
        '''
            Execute one step given by the agent.

            Determine if episode needs to be terminated based on success or failure conditions
            If episode is terminated in an actions < the number of action_repeats, append the
            last state (the termination state) to the returned concatenated states to return
            a fully shaped state
        '''
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities

            if self.goal_reached():
                done = True
                self.success = True

            if self.out_of_bounds():
                done = True

            reward += self.get_reward(done)
            pose_all.append(self.sim.pose)
            if done:
                missing = self.action_repeat - len(pose_all)
                pose_all.extend([pose_all[-1]] * missing)
                break

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """
            Reset the sim to start a new episode.

            Reset distance to target
            Reset success
        """
        self.sim.reset()
        self.distance_to_target = self.init_distance
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        self.success = False
        return state