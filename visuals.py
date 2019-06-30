from IPython.display import display, HTML
import os
import io
import base64
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def render_video(path):
    """Show a video at `path` within IPython Notebook
    """
    if not os.path.isfile(path):
        raise NameError("Cannot access: {}".format(path))

    video = io.open(path, 'r+b').read()
    encoded = base64.b64encode(video)

    display(HTML(
        data="""
        <video alt="test" controls>
        <source src="data:video/mp4;base64,{0}" type="video/mp4" />
        </video>
        """.format(encoded.decode('ascii'))
    ))



def plot_mountaincar_behavior(results):
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    # Set the title
    fig.suptitle("Behavior of Mountaincar agent", fontsize=16, y=1.05)

    ax.plot(results['Position'], label='Position')
    ax.plot(results['Action'], label='Action')
    ax.legend()

def plot_quad_behavior(results, agent, task):

    num_episodes = results['Episode'].max()

    #results = results[results['Episode'] > num_episodes-100]

    best_ep = results.groupby(['Episode'])[['Reward']].sum().idxmax()[0]
    ep = results.loc[results.Episode == best_ep]

    f3, ax3 = plt.subplots(3, 2, figsize=(14, 14))
    f3.suptitle("Behavior of {} agent on the task {} of best episode ({:03.2f} at ep {})".format(agent, task, ep['Reward'].sum(), best_ep), fontsize=16,
                y=1.05)

    ax3[0,0].plot(ep['time'], ep['x'], label='X')
    ax3[0,0].plot(ep['time'], ep['y'], label='Y')
    ax3[0,0].plot(ep['time'], ep['z'], label='Z')
    ax3[0,0].set_title('{}: \nPosition during best episode'.format(agent))
    ax3[0,0].legend()

    ax3[0,1].plot(ep['time'], ep['x_velocity'], label='X_v')
    ax3[0,1].plot(ep['time'], ep['y_velocity'], label='Y_v')
    ax3[0,1].plot(ep['time'], ep['z_velocity'], label='Z_v')
    ax3[0,1].set_title('{}: \nVelocity during best episode'.format(agent))
    ax3[0,1].legend()

    ax3[1,0].plot(ep['time'], ep['phi'], label='phi')
    ax3[1,0].plot(ep['time'], ep['theta'], label='zeta')
    ax3[1,0].plot(ep['time'], ep['psi'], label='psi')
    ax3[1,0].set_title('{}: \nEuler angles during best episode'.format(agent))
    ax3[1,0].legend()

    ax3[1,1].plot(ep['time'], ep['phi_velocity'], label='phi_v')
    ax3[1,1].plot(ep['time'], ep['theta_velocity'], label='zeta_v')
    ax3[1,1].plot(ep['time'], ep['psi_velocity'], label='psi_v')
    ax3[1,1].set_title('{}: \nAngle velocity during best episode'.format(agent))
    ax3[1,1].legend()

    ax3[2,0].plot(ep['time'], ep['rotor1'], label='Rotor 1')
    ax3[2,0].plot(ep['time'], ep['rotor2'], label='Rotor 2')
    ax3[2,0].plot(ep['time'], ep['rotor3'], label='Rotor 3')
    ax3[2,0].plot(ep['time'], ep['rotor4'], label='Rotor 4')
    ax3[2,0].set_title('{}: \nRotor speeds during best episode'.format(agent))
    ax3[2,0].legend()

    f2 = plt.figure(figsize=(14, 8))
    ax2 = f2.add_subplot(111, projection='3d')
    ax2.set_title('{}: \nFlight path'.format(agent))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.scatter(ep['x'], ep['y'], ep['z'])
    ax2.set_xlim([-10, 10])
    ax2.set_ylim([-10, 10])


def plot_training_rewards(results, agent, task, n, verbose=False, reward_function=None):
    # Create figure
    f1, ax = plt.subplots(1, 2, figsize=(14, 5))
    # Set the title
    f1.suptitle("Reward earned by the {} agent on the task: {}".format(agent, task), fontsize=16, y=1.05)

    episode_rewards = results.groupby(['Episode'])[['Reward']].sum()

    # In the left graph, plot the total reward for each episode and the running mean.
    average_reward = episode_rewards.rolling(n).mean()  # running_mean of n

    ax[0].plot(average_reward, label='Running Average Reward (n={})'.format(n))
    ax[0].plot(episode_rewards, color='grey', alpha=0.5, label='Total Reward per Episode')
    ax[0].set_title('{}: \nTotal Reward per Episode in {}'.format(agent, task))
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Reward')
    ax[0].legend()

    episode_rewards = episode_rewards[-n:]
    average_reward = average_reward[-n:]

    ax[1].plot(average_reward, label='Running Average Reward (n={})'.format(n))
    ax[1].plot(episode_rewards, color='grey', alpha=0.5, label='Total Reward per Episode')
    ax[1].set_title('{}: \nTotal Reward per Episode in {} over final {} episodes'.format(agent, task, n))
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Reward')
    ax[1].legend()



    if verbose:
        f2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
        f2.suptitle("Experiment parameters of {} agent on the task: {}".format(agent, task), fontsize=18, y=1.05)

        episode_p = results.groupby(['Episode'])[['p']].max()

        ax2[0].plot(episode_p, label='p')
        ax2[0].set_title('{}: \nExploration/Exploitation Trade off'.format(agent))
        ax2[0].set_xlabel('Episode')
        ax2[0].set_ylabel('p')
        ax2[0].legend()

        if reward_function is not None:
            r = [reward_function(x) for x in range(0,100)]
            episode_p = results.groupby(['Episode'])[['p']].max()

            ax2[1].plot(r, label='Reward function')
            ax2[1].set_title('{}: \nReward function from 0 to 100'.format(agent))
            ax2[1].set_xlabel('Measure')
            ax2[1].set_ylabel('Reward')
            ax2[1].legend()


def plot_quad_training_action(results, agent, task):
    f, axarr = plt.subplots(2, 2, figsize=(14, 5))
    f.suptitle("Action history of {} agent on the task: {}".format(agent, task), fontsize=18, y=1.05)

    r1_mean = results.groupby(['Episode'])[['rotor1']].mean()
    r2_mean = results.groupby(['Episode'])[['rotor2']].mean()
    r3_mean = results.groupby(['Episode'])[['rotor3']].mean()
    r4_mean = results.groupby(['Episode'])[['rotor4']].mean()

    r1_min = results.groupby(['Episode'])[['rotor1']].min()
    r2_min = results.groupby(['Episode'])[['rotor2']].min()
    r3_min = results.groupby(['Episode'])[['rotor3']].min()
    r4_min = results.groupby(['Episode'])[['rotor4']].min()

    r1_max = results.groupby(['Episode'])[['rotor1']].max()
    r2_max = results.groupby(['Episode'])[['rotor2']].max()
    r3_max = results.groupby(['Episode'])[['rotor3']].max()
    r4_max = results.groupby(['Episode'])[['rotor4']].max()

    r1_std = results.groupby(['Episode'])[['rotor1']].std()
    r2_std = results.groupby(['Episode'])[['rotor2']].std()
    r3_std = results.groupby(['Episode'])[['rotor3']].std()
    r4_std = results.groupby(['Episode'])[['rotor4']].std()

    axarr[0, 0].plot(r1_mean, label='Rotor 1')
    axarr[0, 0].plot(r2_mean, label='Rotor 2')
    axarr[0, 0].plot(r3_mean, label='Rotor 3')
    axarr[0, 0].plot(r4_mean, label='Rotor 4')
    axarr[0, 0].set_title('Rotor Mean')
    axarr[0, 0].legend()

    axarr[0, 1].plot(r1_min, label='Rotor 1')
    axarr[0, 1].plot(r2_min, label='Rotor 2')
    axarr[0, 1].plot(r3_min, label='Rotor 3')
    axarr[0, 1].plot(r4_min, label='Rotor 4')
    axarr[0, 1].set_title('Rotor Min')
    axarr[0, 1].legend()

    axarr[1, 0].plot(r1_max, label='Rotor 1')
    axarr[1, 0].plot(r2_max, label='Rotor 2')
    axarr[1, 0].plot(r3_max, label='Rotor 3')
    axarr[1, 0].plot(r4_max, label='Rotor 4')
    axarr[1, 0].set_title('Rotor Max')
    axarr[1, 0].legend()

    axarr[1, 1].plot(r1_std, label='Rotor 1')
    axarr[1, 1].plot(r2_std, label='Rotor 2')
    axarr[1, 1].plot(r3_std, label='Rotor 3')
    axarr[1, 1].plot(r4_std, label='Rotor 4')
    axarr[1, 1].set_title('Rotor Stdev')
    axarr[1, 1].legend()

    f2, ax = plt.subplots(1, 1, figsize=(14, 3))

    ax.hist(results['rotor1'], label='Rotor 1', alpha=0.5, bins=20)
    ax.hist(results['rotor2'], label='Rotor 2', alpha=0.5, bins=20)
    ax.hist(results['rotor3'], label='Rotor 3', alpha=0.5, bins=20)
    ax.hist(results['rotor4'], label='Rotor 4', alpha=0.5, bins=20)
    ax.legend()
    ax.set_title('Distribution of actions over entire training')
