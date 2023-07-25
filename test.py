import numpy as np
import math
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成随机数据点
# x = np.random.rand(50)
# y = np.random.rand(50)
# z = np.sin(x*y)
#
# # 对数据点进行排序
# idx = np.argsort(x)
# x = x[idx]
# y = y[idx]
# z = z[idx]
#
# # 进行三次样条插值拟合
# f = interp2d(x, y, z, kind='cubic')
#
# # 生成插值结果的网格数据
# xnew, ynew = np.mgrid[0:1:100j, 0:1:100j]
# znew = f(xnew, ynew)
#
# # 可视化插值结果
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xnew, ynew, znew)
# plt.show()

# theta = np.pi/6
# T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
# action = np.array([10, 2, 25])
# action = action.reshape((-1, 1))
# print(action.shape)
# print(action)
# m = T @ action
# print(m)
# print(m.shape)
# print(10*m)

# env = gym.make('CartPole-v1')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break
# env.close()

# theta = np.array([0.354])
# T = np.array([[math.cos(theta), -math.sin(theta), 0.], [math.sin(theta), math.cos(theta), 0.], [0., 0., 1.]])
# print(type(math.cos(theta)))
# print(T)
from USVmodel import USVTracking
path_line = [[10, i] for i in range(1000)]
env = USVTracking(path_line)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(obs_dim, act_dim)



