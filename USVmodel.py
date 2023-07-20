import gym
import numpy as np
import math
from gym import spaces


class USVTracking(gym.Env):
    """
    ###
    1.USV模型：USV由布置在前后左右的四个螺旋桨推力驱动，动力学模型见陈鑫源学长论文，输入4×1的推力向量，输出位置和速度的六元组（x,y,theta,u,v,omega）
    3.采样时间 T = 0.5s
    2.track：定义规则曲线或由 path_ref_gen.py 生成 waypoints，路径也是观测量
    ###
    X=(x,y,theta), V=(u,v,omega), p=waypoint
    observation space:
    X_t, V_t, p_ref_t, p_ref_t-1, p_ref_t+1
    ###
    action space:
    F=(f1,f2,f3,f4)T
    ###
    reward:
    r = k_a*exp(-epsilon/epsilon_0)+k_b*exp(-(theta - arctan(y_ref'/x_ref'))**2)
    ###
    初始状态
    USV初始位置随机在参考轨迹起始点为圆心半径为 R 的圆内
    初始速度和初始角为 0
    """

    def __init__(self, path_ref: list):
        self.sample_time = 0.5  # 采样时间
        self.a = 0.45  # USV宽度
        self.b = 0.9  # usv长度
        self.path_ref = path_ref  # 参考路径
        self.noise = np.zeros((4, 1))  # 噪声
        # reward 系数
        self.k_a = 3.0
        self.k_b = 1.0
        # 系统惯量
        self.m11 = 172
        self.m22 = 188
        self.m33 = 24
        # 阻尼系数
        self.Xu = 38
        self.Yv = 168
        self.Nr = 16
        # 模型系数阵B, M, D
        self.B = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [self.a / 2, -self.a / 2, self.b / 2, -self.b / 2]])
        self.M = np.diag([self.m11, self.m22, self.m33])
        self.Minv = np.linalg.inv(self.M)
        self.D = np.diag([self.Xu, self.Yv, self.Nr])
        # 状态约束
        self.min_action = np.array([-500, -500, -500, -500], dtype=np.float32)
        self.max_action = np.array([500, 500, 500, 500], dtype=np.float32)
        self.max_vector_x = np.inf  # vector_x=x_p-x
        self.max_vector_y = np.inf  # vector_y=y_p-y
        self.max_theta = np.pi
        self.max_u = 200
        self.max_v = 200
        self.max_omega = 10 * np.pi
        self.max_gama_ref = np.pi  # 参考点轨迹一阶差分的arctan2函数
        self.max_gama_ref_2order = np.pi  # 参考点轨迹二阶差分的arctan2函数
        self.low_state = np.array(
            [-self.max_vector_x, -self.max_vector_y, -self.max_theta, -self.max_u, -self.max_v, -self.max_omega,
             -self.max_gama_ref, -self.max_gama_ref_2order], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_vector_x, self.max_vector_y, self.max_theta, self.max_u, self.max_v, self.max_omega,
             self.max_gama_ref, self.max_gama_ref_2order], dtype=np.float32
        )
        # 初始化参数，在reset()中赋值
        self.state = np.zeros((8, 1))
        self.state_position = np.zeros((3, 1))
        self.state_velocity = np.zeros((3, 1))
        self.index_ref = 0
        self.epsilon0 = 10
        # 速度空间

        # 动作空间和观测空间
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

    def step(self, action: np.array):
        state = self.state
        state_position = self.state_position
        state_velocity = self.state_velocity
        theta = float(state_position[2])
        u = float(state_velocity[0])
        v = float(state_velocity[1])

        # 计算运动学和动力学模型的系数矩阵T，M，B，C，D，其中M，B，D为常系数，由类的self属性给出
        T = np.array([[math.cos(theta), -math.sin(theta), 0.], [math.sin(theta), math.cos(theta), 0.], [0., 0., 1.]])
        C = np.array([[0, 0, -self.m22 * v], [0, 0, self.m11 * u],
                      [self.m22 * v, -self.m11 * u, 0]])
        action = action.reshape((-1, 1))
        state_position_reshape = state_position.reshape((-1, 1))
        state_velocity_reshape = state_velocity.reshape((-1, 1))

        # 离散的运动学和动力学模型
        self.state_position = state_position_reshape + self.sample_time * T @ state_velocity_reshape
        self.state_velocity = state_velocity_reshape + self.sample_time * (self.Minv @ self.B @ (action + self.noise)
                                                                           - self.Minv @ (C + self.D)
                                                                           @ state_velocity_reshape)

        # 搜索距离船最近点
        index_ref = self.index_ref
        index_count = max(self.index_ref - 20, 0)
        start = index_count
        end = min(self.index_ref + 20, len(self.path_ref))
        min_dist = 100
        for waypoint in self.path_ref[start: end]:
            waypoint_array = np.array(waypoint)
            dist = np.linalg.norm(x=waypoint_array - self.state_position[0:2])
            if dist < min_dist:
                min_dist = dist
                self.index_ref = index_count
            index_count += 1

        # 终止判断
        terminated = bool(self.index_ref == len(self.path_ref))

        # 下一时刻反馈状态量计算
        x = float(self.state_position[0])
        y = float(self.state_position[1])
        theta = float(self.state_position[2])
        u = float(self.state_velocity[0])
        v = float(self.state_velocity[1])
        omega = float(self.state_velocity[2])
        # vector_x = x_p - x; vector_y = y_p - y
        vector_x = self.path_ref[self.index_ref][0] - x
        vector_y = self.path_ref[self.index_ref][1] - y
        # 一阶中心差分dealt_y, dealt_x，arctan2(y'/x') = gama_p 为路径切向角
        dealt_y = self.path_ref[self.index_ref + 1][1] - self.path_ref[self.index_ref - 1][1]
        dealt_x = self.path_ref[self.index_ref + 1][0] - self.path_ref[self.index_ref - 1][0]
        gama_ref = math.atan2(dealt_y, dealt_x)
        # 二阶中心差分，gama_ref_2order 为路径切向角变化率
        dealt_y_2order = self.path_ref[self.index_ref + 1][1] - 2 * self.path_ref[self.index_ref][1] + \
                         self.path_ref[self.index_ref - 1][1]
        dealt_x_2order = self.path_ref[self.index_ref + 1][0] - 2 * self.path_ref[self.index_ref][0] + \
                         self.path_ref[self.index_ref - 1][0]
        gama_ref_2order = math.atan2(dealt_y_2order, dealt_x_2order)
        self.state = np.array([vector_x, vector_y, theta, u, v, omega, gama_ref, gama_ref_2order], dtype=np.float32)

        # reward definition
        epsilon = -vector_y * math.cos(gama_ref) + vector_x * math.sin(gama_ref)
        gama_usv = math.atan2(u * math.sin(theta) + v * math.cos(theta), u * math.cos(theta) - v * math.sin(theta))
        reward = self.k_a * math.exp(-abs(epsilon / self.epsilon0)) + self.k_b * math.exp(
            -np.square(gama_ref - gama_usv))

        # 对越轨的状态进行限制和惩罚
        if epsilon > 100 or not self.observation_space.contains(self.state):
            self.state_position = state_position
            self.state_velocity = state_velocity
            self.state = state
            self.index_ref = index_ref
            reward = 0.
            print('overshoot!')

        print(reward, self.state)
        return self.state, reward, terminated, False, {}

    def reset(self):
        R = 5  # USV起点与路径起点的最大距离，在该圆内平均采样
        theta_sample = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, R ** 2))
        x_start = self.path_ref[0][0] + r * np.cos(theta_sample)
        y_start = self.path_ref[0][1] + r * np.sin(theta_sample)
        theta_start = np.random.uniform(-np.pi, np.pi)
        self.state_position = np.array([x_start, y_start, theta_start], dtype=np.float32).reshape(3, 1)
        self.state_velocity = np.array([0., 0., 0.], dtype=np.float32).reshape(3, 1)

        # vector_x = x_p - x; vector_y = y_p - y
        vector_x = self.path_ref[1][0] - x_start
        vector_y = self.path_ref[1][1] - y_start
        # 一阶中心差分dealt_y, dealt_x，arctan2(y'/x') = gama_p 为路径切向角
        dealt_y = self.path_ref[2][1] - self.path_ref[0][1]
        dealt_x = self.path_ref[2][0] - self.path_ref[0][0]
        gama_ref = np.arctan2(dealt_y, dealt_x)
        # 二阶中心差分，gama_ref_2order 为路径切向角变化率
        dealt_y_2order = self.path_ref[2][1] - 2 * self.path_ref[1][1] + \
                         self.path_ref[0][1]
        dealt_x_2order = self.path_ref[2][0] - 2 * self.path_ref[1][0] + \
                         self.path_ref[0][0]
        gama_ref_2order = np.arctan2(dealt_y_2order, dealt_x_2order)
        self.index_ref = 1
        self.epsilon0 = -vector_y * np.cos(gama_ref) + vector_x * np.sin(gama_ref)
        self.state = np.array([vector_x, vector_y, theta_start, 0., 0., 0., gama_ref, gama_ref_2order],
                              dtype=np.float32)
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass
