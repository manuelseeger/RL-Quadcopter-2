import csv
import math
import os
import sys

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

from agents.agent import DDPG_Agent
from agents.policy_search import PolicySearch_Agent
from agents.random_binary_agent import Random_Binary_Agent
from collections import deque
from tasks import TakeOff_Task

ex = Experiment()
ex.observers.append(MongoObserver.create(db_name='sacred'))

@ex.config
def config():

    # Noise process
    exploration_mu = 0.
    exploration_theta = 0.15
    exploration_sigma = 0.2

    # Replay memory
    buffer_size = 100000
    batch_size = 128

    # Algorithm parameters
    gamma = 0.99   # discount factor
    tau = 0.01  # for soft update of target parameters

    # Experiment
    num_episodes = 1000
    runtime = 5.
    success_mem_len = 10
    minimum_successes = 9

    # Task parameters
    init_velocities = np.array([0.1, 0.1, 0.1])         # initial velocities
    init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
    file_output = 'data.txt'                         # file name for saved results
    init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
    action_low = 10
    action_high = 900
    action_size = 4
    action_repeat = 1
    target_pos = np.array([0., 0., 50.])

    # experiment logging parameters
    n_mean = 10
    test_log_file_name = 'test_log.txt'
    write_train_log = False

    # which agent to run
    agents = ['DDPG', 'Policy_Search', 'Random_Binary']
    agent_type = agents[0]

    success_distance=2


@ex.capture
def init(target_pos, init_pose, init_angle_velocities, init_velocities, runtime, action_low, action_high, agent_type,
            action_repeat, action_size, success_mem_len,
            gamma=0.9, tau=0.1, buffer_size=100000, batch_size=128, exploration_mu=0,
            exploration_theta=0.15, exploration_sigma=0.2, success_distance=1):

    task = TakeOff_Task(target_pos=target_pos, init_pose=init_pose,
                init_angle_velocities=init_angle_velocities, init_velocities=init_velocities,
                runtime=runtime)

    task.configure(action_repeat=action_repeat, action_low=action_low, action_high=action_high, action_size=action_size,
                   target_pos=target_pos, init_velocities=init_velocities, init_angle_velocities=init_angle_velocities,
                   init_pose=init_pose, success_distance=success_distance)

    if agent_type == 'DDPG':
        agent = DDPG_Agent(task)
        agent.configure(gamma, tau, buffer_size, batch_size, exploration_mu, exploration_theta, exploration_sigma)
    if agent_type == 'Policy_Search':
        agent = PolicySearch_Agent(task)
    if agent_type == 'Random_Binary':
        agent = Random_Binary_Agent(task)
        agent.configure(success_mem_len)


    return task, agent

@ex.capture
def train(_run, task, agent, num_episodes, n_mean, write_train_log, success_mem_len, minimum_successes):

    rewards = np.array([])
    successes = deque([], maxlen=success_mem_len)

    if write_train_log:
        f = open('rewards_log.txt', 'w')

    for i_episode in range(1, num_episodes + 1):
        state = agent.reset_episode()  # start a new episode
        total_reward = 0.

        p = 1 - (i_episode / (num_episodes * 0.5)) ** 0.5  # exploration / exploitation trade off
        p = max(p, 0)

        actions = []

        while True:
            action = agent.act(state, p)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            total_reward += reward

            actions.append(action)

            if done:

                actions = np.array(actions).reshape(-1, 4)

                means = actions.mean(axis=0)
                stds = actions.std(axis=0)
                mins = actions.min(axis=0)
                maxs = actions.max(axis=0)

                print("\rEpisode = {:4d}, Reward = {:8.4f}, {:7} ({:.2f}), Rotors mean: {:03.0f} {:03.0f} {:03.0f} {:03.0f}".format(
                        i_episode, total_reward, ('Success' if agent.task.success else 'Fail ({})'.format(agent.task.outcome)),
                        agent.task.distance_to_target, means[0], means[1], means[2], means[3]))
                print("\r{:>60}std:  {:03.0f} {:03.0f} {:03.0f} {:03.0f}".format('',
                    stds[0], stds[1], stds[2], stds[3], end=""
                ))
                print("\r{:>60}min:  {:03.0f} {:03.0f} {:03.0f} {:03.0f}".format('',
                    mins[0], mins[1], mins[2], mins[3], end=""
                ))
                print("\r{:>60}max:  {:03.0f} {:03.0f} {:03.0f} {:03.0f}".format('',
                    maxs[0], maxs[1], maxs[2], maxs[3], end=""
                ))

                if write_train_log:
                    f.writelines(str(total_reward) + '\n')
                    f.flush()

                rewards = np.append(rewards, total_reward)
                n = n_mean if n_mean < len(rewards) else len(rewards)
                moving_average = np.sum(rewards[-n:])/n
                _run.log_scalar('Reward', total_reward, i_episode)
                _run.log_scalar('Distance', agent.task.distance_to_target)
                _run.log_scalar('Past {:d} episode mean reward'.format(n_mean), moving_average, i_episode)
                total_reward = 0

                successes.append(agent.task.success)

                break

        if sum(successes) >= minimum_successes:
            break
        sys.stdout.flush()

@ex.capture
def test(_run, agent, task, test_log_file_name, init_pose):
    done = False
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    results = {x: [] for x in labels}
    # Run the simulation, and save the results.
    with open(test_log_file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)

        state = agent.reset_episode()

        while True:
            rotor_speeds = agent.act(state)
            state, _, done = task.step(rotor_speeds)
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
            _run.log_scalar('X', task.sim.pose[0])
            _run.log_scalar('Y', task.sim.pose[1])
            _run.log_scalar('Z', task.sim.pose[2])

            _run.log_scalar('phi', task.sim.pose[3])
            _run.log_scalar('theta', task.sim.pose[4])
            _run.log_scalar('psi', task.sim.pose[5])

            _run.log_scalar('A1', rotor_speeds[0])
            _run.log_scalar('A2', rotor_speeds[1])
            _run.log_scalar('A3', rotor_speeds[2])
            _run.log_scalar('A4', rotor_speeds[3])

            _run.log_scalar('X-v', task.sim.v[0])
            _run.log_scalar('Y-v', task.sim.v[1])
            _run.log_scalar('Z-v', task.sim.v[2])

            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            if done:
                break
@ex.automain
def main(_run):
    task, agent = init()

    train(_run, task, agent)

    #os.mkdir(os.path.join('runs', str(_run._id)))
    #agent.actor_target.model.save(os.path.join('runs', str(_run._id), 'actor.h5'))
    #agent.critic_target.model.save(os.path.join('runs', str(_run._id), 'critic.h5'))

    #ex.add_artifact(os.path.join('runs', str(_run._id), 'actor.h5'))
    #ex.add_artifact(os.path.join('runs', str(_run._id), 'critic.h5'))

    test(_run, agent, task)
