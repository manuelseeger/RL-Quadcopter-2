import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        self.success = False

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.distance_to_target = 0.
        self.init_distance = np.linalg.norm(self.target_pos - self.sim.init_pose[:3])

    def configure(self, action_repeat, action_low, action_high, action_size, target_pos,
                  init_velocities, init_angle_velocities, init_pose):
        self.action_low = action_low
        self.action_high = action_high
        self.action_repeat = action_repeat
        self.action_size = action_size
        self.target_pos = target_pos

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        self.distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])

        reward = 1 - (self.distance_to_target / (self.init_distance * 2)) ** 2

        #reward = 1-np.tanh(self.target_pos[2] - self.sim.pose[2])
        #reward = np.tanh(1 - 0.003 * (abs(self.sim.pose[2] - self.target_pos[2]))).sum()

        reward = reward / self.action_repeat

        if done and self.goal_reached():
            reward += 100
        if done and self.out_of_bounds():
            reward -= 20
        if done and not self.goal_reached() and self.sim.runtime > self.sim.time:
            reward -= 50

        return reward

    def goal_reached(self):
        return self.distance_to_target < 1

    def out_of_bounds(self):
        return self.distance_to_target > self.init_distance * 2

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
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
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.distance_to_target = self.init_distance
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        self.success = False
        return state