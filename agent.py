import parl
import paddle
import numpy as np


class Agent(parl.Agent):
    def __init__(self, algorithm, act_dim, expl_noise=0.1):
        assert isinstance(act_dim, int)
        super(Agent, self).__init__(algorithm)

        self.act_dim = act_dim
        self.expl_noise = expl_noise

        # 注意：最开始先同步self.model和self.target_model的参数.
        self.alg.sync_target(decay=0)

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        action_numpy = self.predict(obs)
        action_noise = np.random.normal(0, self.expl_noise, size=self.act_dim)
        action = (action_numpy + action_noise).clip(-1, 1)
        return action

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action = self.alg.predict(obs)
        action_numpy = action.cpu().numpy()[0]
        action_numpy = action_numpy.clip(-1, 1)
        return action_numpy

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 根据训练数据更新一次模型参数
        """
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        action = paddle.to_tensor(action, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        return critic_loss, actor_loss
