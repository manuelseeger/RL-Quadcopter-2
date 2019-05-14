import numpy as np
from physics_sim import PhysicsSim
import math

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
        self.action_low = 10
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 50.])

        if init_pose is not None:
            self.init_edist = np.linalg.norm(self.target_pos - self.sim.init_pose[0:3])
        else:
            self.init_edist = 0

        self.success = False

        self.distance_to_target = self.init_edist

    def goal_reached(self):
        return self.distance_to_target < 2

    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""

        self.distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[0:3])

        reward = 1 - (self.distance_to_target / (self.init_edist * 2)) ** .3

        if self.goal_reached():
            reward += 50

        reward = reward / self.action_repeat

        if self.sim.done and self.sim.runtime > self.sim.time:
            reward = -20

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds)
            pose_all.append(self.sim.pose)
            if done:
                missing = self.action_repeat - len(pose_all)
                if missing > 0:
                    pose_all.extend([pose_all[-1]] * missing)
                break

        next_state = np.concatenate(pose_all)

        # stop once target heigh reached
        if self.goal_reached():
            done = True
            self.success = True

        return next_state, reward, done

    def reset(self, init_pose=None):
        """Reset the sim to start a new episode."""
        if init_pose is not None:
            self.sim.init_pose = init_pose
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        self.success = False
        return state