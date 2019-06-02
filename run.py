import sys
from agents.agent import DDPG_Agent
import csv
import numpy as np
from task_playground import Task
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Modify the values below to give the quadcopter a different starting position.
runtime = 5.                                     # time limit of the episode
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
file_output = 'data.txt'                         # file name for saved results
init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose


num_episodes = 1000
target_pos = np.array([0., 0., 50.])
task = Task(target_pos=target_pos, init_pose=init_pose)
agent = DDPG_Agent(task)

rewards = []

f = open('rewards_log.txt', 'w')

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
#            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format())
#                 i_episode, agent.score, agent.best_score, agent.noise_scale), end="")  # [debug]
            print("\rEpisode = {:4d}, Reward = {:4f}".format(i_episode, reward), end="")

            rewards.append(reward)
            f.writelines(str(reward) + '\n')
            f.flush()

            break
    sys.stdout.flush()

done = False
labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
results = {x : [] for x in labels}

# Run the simulation, and save the results.
with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)

    state = agent.reset_episode(init_pose)

    while True:
        rotor_speeds = agent.act(state)
        state, _, done = task.step(rotor_speeds)
        to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
        for ii in range(len(labels)):
            results[labels[ii]].append(to_write[ii])
        writer.writerow(to_write)
        if done:
            break


plt.plot(results['time'], results['x'], label='x')
plt.plot(results['time'], results['y'], label='y')
plt.plot(results['time'], results['z'], label='z')


plt.show()

#fig = plt.figure(figsize = (14,8))
#ax = fig.add_subplot(111, projection='3d')

#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')

#ax.scatter(results['x'], results['y'], results['z'])

'''

fig1 = plt.figure(figsize=(14,8))
ax1 = fig1.add_subplot(111)
line1, = ax1.plot(rewards)
fig2 = plt.figure(figsize=(14,8))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.scatter(results['x'], results['y'], results['z'])
fig2.show()


fig3 = plt.figure(figsize=(14,8))
ax3 = fig3.add_subplot(111)
ax3.plot(results['time'], results['x'], label='x')
ax3.plot(results['time'], results['y'], label='y')
ax3.plot(results['time'], results['z'], label='z')
ax3.legend()
fig3.show()
'''

