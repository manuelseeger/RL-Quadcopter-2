import numpy as np

start = np.array([0., 0., 10., 0., 0., 0.])
target = target_pos = np.array([0., 0., 50.])

edist = np.linalg.norm(target - start[:3])

print(edist)



'''
N = [0,1]

plt.ion()
fig = plt.figure(figsize=(14,8))
ax1 = fig.add_subplot(111)

x = np.arange(len(N))
y = np.array(N)

line1, = ax1.plot(x, y)

ax1.set_xlim(0, 100)
ax1.set_ylim(-50, +50)

plt.draw()

N.append(1)
N.append(2)
N.append(3)

x = np.arange(len(N))
y = np.array(N)

line1.set_data(x, y)

ax1.relim()
ax1.autoscale()
plt.draw()
plt.pause(0.02)
plt.show()
'''