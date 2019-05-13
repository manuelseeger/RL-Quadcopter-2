from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
from agents.agent import  DDPG_Agent
from task import Task
import sys
import os
import csv

ex = Experiment()
ex.observers.append(MongoObserver.create(db_name='sacred'))

@ex.config
def config():
    gamma = 0.99
    tau = 0.5

    # Noise process
    exploration_mu = 0.1
    exploration_theta = -0.15
    exploration_sigma = 0.2

    # Replay memory
    buffer_size = 100000
    batch_size = 64

    # Algorithm parameters
    gamma = 0.99  # discount factor
    tau = 0.1  # for soft update of target parameters

    # Experiment
    num_episodes = 1000
    runtime = 5.

    # Task parameters
    init_velocities = np.array([0., 0., 0.])         # initial velocities
    init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
    file_output = 'data.txt'                         # file name for saved results
    init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose

    target_pos = np.array([0., 0., 50.])

    # experiment logging parameters
    window = 100
    test_log_file_name = 'test_log.txt'
    write_train_log = False


@ex.capture
def init(target_pos, init_pose, init_angle_velocities, init_velocities,
            gamma=0.9, tau=0.1, buffer_size=100000, batch_size=128, exploration_mu=0,
            exploration_theta=0.15, exploration_sigma=0.2):
    task = Task(target_pos=target_pos, init_pose=init_pose,
                init_angle_velocities=init_angle_velocities, init_velocities=init_velocities)
    agent = DDPG_Agent(task)
    agent.configure(gamma, tau, buffer_size, batch_size, exploration_mu, exploration_theta, exploration_sigma)

    return task, agent

@ex.capture
def train(_run, task, agent, num_episodes, window, write_train_log):

    rewards = np.array([])

    if write_train_log:
        f = open('rewards_log.txt', 'w')

    for i_episode in range(1, num_episodes + 1):
        state = agent.reset_episode()  # start a new episode
        total_reward = 0.
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                if total_reward == float('nan'):
                    total_reward = 0.
                print("\rEpisode = {:4d}, Reward = {:4f}".format(i_episode, total_reward), end="")
                if write_train_log:
                    f.writelines(str(total_reward) + '\n')
                    f.flush()

                rewards = np.append(rewards, total_reward)
                n = window if window < len(rewards) else len(rewards)
                moving_average = np.sum(rewards[-n:])/n
                _run.log_scalar('Reward', total_reward, i_episode)
                _run.log_scalar('Past {:d} episode mean reward'.format(window), moving_average, i_episode)
                total_reward = 0
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

        state = agent.reset_episode(init_pose)

        while True:
            rotor_speeds = agent.act(state)
            state, _, done = task.step(rotor_speeds)
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
            _run.log_scalar('Z', task.sim.pose[2])
            _run.log_scalar('X', task.sim.pose[0])
            _run.log_scalar('Y', task.sim.pose[1])
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            if done:
                break
@ex.automain
def main(_run):
    task, agent = init()

    train(_run, task, agent)

    os.mkdir(os.path.join('runs', str(_run._id)))
    #agent.actor_target.model.save(os.path.join('runs', str(_run._id), 'actor.h5'))
    #agent.critic_target.model.save(os.path.join('runs', str(_run._id), 'critic.h5'))

    #ex.add_artifact(os.path.join('runs', str(_run._id), 'actor.h5'))
    #ex.add_artifact(os.path.join('runs', str(_run._id), 'critic.h5'))

    test(_run, agent, task)
