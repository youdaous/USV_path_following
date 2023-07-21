import numpy as np

from USVmodel import USVTracking
import matplotlib.pyplot as plt
# import numpy as np

path_line = [[10, i] for i in range(1000)]
query = []
count = 0
env = USVTracking(path_line)
env.reset()
print(env.state_position)
print(env.state_velocity)
while True:
    print(count)
    # action = env.action_space.sample()
    action = np.array([100, -100, 100, -100])
    observation, reward, done, _, _ = env.step(action)
    point = env.state_position[0:2]
    query.append(list(point))
    count += 1
    if done or count > 1000:
        break

print()
m = [dot[0] for dot in query]
n = [dot[1] for dot in query]
x = [po[0] for po in path_line]
y = [po[1] for po in path_line]
plt.plot(n, m)
plt.plot(y, x)
plt.show()


