import matplotlib.pyplot as plt
import numpy as np
import math
# from scipy.interpolate import make_interp_spline

option = [0, 1, 2, 3, 4]
prob = [0., 0.1, 0.8, 0.1, 0.]
origin = np.array([0, 0])
point = origin
query = []
x = []
y = []
inter = 1
move0 = np.array([0, 1]) * inter
move1 = np.array([math.sqrt(0.5), math.sqrt(0.5)]) * inter
move2 = np.array([1, 0]) * inter
move3 = np.array([math.sqrt(0.5), math.sqrt(0.5)]) * inter
move4 = np.array([0, -1]) * inter
trans0 = np.array([[0, 1], [-1, 0]])
trans1 = np.array([[math.sqrt(0.5), math.sqrt(0.5)], [-math.sqrt(0.5), math.sqrt(0.5)]])
trans2 = np.array([[1, 0], [0, 1]])
trans3 = np.array([[math.sqrt(0.5), -math.sqrt(0.5)], [math.sqrt(0.5), math.sqrt(0.5)]])
trans4 = np.array([[0, -1], [1, 0]])
vector = trans2
for _ in range(200):
    direct = np.random.choice(option, size=1, p=prob)

    if direct == 0:
        point = point + move0 @ vector
        vector = trans0 @ vector
    elif direct == 1:
        point = point + move1 @ vector
        vector = trans1 @ vector
    elif direct == 2:
        point = point + move2 @ vector
        vector = trans2 @ vector
    elif direct == 3:
        point = point + move3 @ vector
        vector = trans3 @ vector
    else:
        point = point + move4 @ vector
        vector = trans4 @ vector
    x.append(point[0])
    y.append(point[1])
    # query
    query.append(list(point))

# print(type(query))
print(len(query))
print(type(len(query)))

# m = [dot[0] for dot in query]
# n = [dot[1] for dot in query]
# # index = 10
# # print(query[index + 1:index + 5])
# plt.figure(1)
plt.plot(x, y)
plt.scatter(x[0], y[0], marker='*')
# plt.figure(2)
# plt.plot(m, n)
plt.show()
