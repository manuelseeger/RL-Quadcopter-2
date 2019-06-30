from agents.agent_mountaincar import DDPG_Agent
from task_mountaincar import Mountain_Task
import sys
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
ex.observers.append(MongoObserver.create(db_name='sacred'))

@ex.config
def config():

    # Noise process
    exploration_mu = 0.
    exploration_theta = 0.05
    exploration_sigma = 0.25

    # Replay memory
    buffer_size = 100000
    batch_size = 256

    # Algorithm parameters
    gamma = 0.99   # discount factor
    tau = 0.001  # for soft update of target parameters

    tau_actor = 0.01
    tau_critic = 0.1

    # Experiment
    num_episodes = 1000

    # Task parameters
    file_output = 'data.txt'                         # file name for saved results

    action_repeat = 3
    throttle = 1

    # experiment logging parameters
    n_mean = 100
    test_log_file_name = 'test_log.txt'
    write_train_log = False

    learn_during_episode = True

    monitor = True

@ex.capture
def init(action_repeat, learn_during_episode, gamma=0.9, tau=0.1, tau_critic=0.5, tau_actor=0.1, buffer_size=100000, batch_size=128, exploration_mu=0,
         exploration_theta=0.15, exploration_sigma=0.2, monitor=False, throttle=1):

    task = Mountain_Task(action_repeat=action_repeat, monitor=False)

    agent = DDPG_Agent(task)
    agent.configure(gamma, tau, tau_critic, tau_actor, buffer_size, batch_size, exploration_mu, exploration_theta, exploration_sigma,
                    learn_during_episode, throttle)

    return task, agent

@ex.capture
def train(_run, task, agent, num_episodes, n_mean, write_train_log):

    rewards = np.array([])

    if write_train_log:
        f = open('rewards_log.txt', 'w')

    for i_episode in range(1, num_episodes + 1):
        state = agent.reset_episode()  # start a new episode
        total_reward = 0.0
        actions = []
        p = 1 - (i_episode / (num_episodes * 0.5)) ** 0.5  # exploration / exploitation trade off
        p = max(p, 0)

        while True:
            action = agent.act(state, p)
            next_state, reward, done, raw_states = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            total_reward += reward

            actions.append(action[0])
            if done:
                print("\rEpisode = {:4d}, Reward = {: 8.4f}, {:7} ({: 2.2f}), Motor: {: 03.4f} / {:03.4f}".format(
                    i_episode, total_reward, ('Success' if task.success else 'Fail'),
                    raw_states[-1][0], np.mean(actions), np.std(actions)))

                if write_train_log:
                    f.writelines(str(total_reward) + '\n')
                    f.flush()

                rewards = np.append(rewards, total_reward)
                n = n_mean if n_mean < len(rewards) else len(rewards)
                moving_average = np.sum(rewards[-n:])/n
                _run.log_scalar('Reward', total_reward, i_episode)
                _run.log_scalar('Distance', raw_states[-1][0])
                _run.log_scalar('Past {:d} episode mean reward'.format(n_mean), moving_average, i_episode)
                total_reward = 0

                break

        if np.sum(rewards[-100:]) / 100 > 90:
            break
        sys.stdout.flush()

@ex.capture
def test(_run, agent, task, test_log_file_name):
    done = False

    # Run the simulation, and save the results.
    state = agent.reset_episode()

    while True:
        action = agent.act(state)
        #task.env.render()
        state, _, done, raw_state = task.step(action)

        _run.log_scalar('P', raw_state[-1][0])

        _run.log_scalar('V', state[1])
        _run.log_scalar('A', action[0])

        if done:
            break

@ex.automain
def main(_run):
    task, agent = init()
    train(_run, task, agent)
    test(_run, agent, task)
