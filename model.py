import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Model(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.actor_model = Actor(obs_dim, act_dim)
        self.critic_model = Critic(obs_dim, act_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class Actor(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        hid_size = 100

        self.l1 = nn.Linear(obs_dim, hid_size)
        self.l2 = nn.Linear(hid_size, act_dim)

    def forward(self, obs):
        hid = F.relu(self.l1(obs))
        means = paddle.tanh(self.l2(hid))
        return means


class Critic(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        hid_size = 100

        self.l1 = nn.Linear(obs_dim + act_dim, hid_size)
        self.l2 = nn.Linear(hid_size, 1)

    def forward(self, obs, act):
        concat = paddle.concat([obs, act], axis=1)
        hid = F.relu(self.l1(concat))
        Q = self.l2(hid)
        return Q
