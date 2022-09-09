import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
EPISODES = 20000
EP_STEPS = 200  # steps for one episode
LR_ACTOR = 0.001  # learning rate of actor
LR_CRITIC = 0.002  # learning rate of critic
GAMMA = 0.9
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # capacity of memory buffer
BATCH_SIZE = 32
RENDER = False
ENV_NAME = 'LunarLanderContinuous-v2'


# DDPG Framework
class ActorNet(nn.Module):  # 定义Actor与Critic的网络结构
    def __init__(self, s_dim, a_dim, fc1_dim, fc2_dim):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(s_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)

        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)

        self.out = nn.Linear(fc2_dim, a_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))  # 激活函数为relu
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.out(x)
        x = torch.tanh(x)  # 利用tanh将值映射到[-1,1]，因为该游戏的动作取值范围为[-1，1]
        return x


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim, fc1_dim, fc2_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(a_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

    def forward(self, s, a):
        x_s = F.relu(self.ln1(self.fc1(s)))
        x_s = self.ln2(self.fc2(x_s))
        x_a = self.fc3(a)
        x = F.relu(x_s + x_a)
        q = self.q(x)

        return q


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # memory buffer
        self.pointer = 0  # buffer的指针（用于存取经验）
        self.actor_loss_list = []
        self.critic_loss_list = []
        # 构建四个网络
        self.actor = ActorNet(s_dim, a_dim, 400, 300)
        self.actor_target = ActorNet(s_dim, a_dim, 400, 300)
        self.critic = CriticNet(s_dim, a_dim, 400, 300)
        self.critic_target = CriticNet(s_dim, a_dim, 400, 300)
        # 构建actor与critic的optimizer（Adam）
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        # 定义训练critic的计算loss函数
        self.loss_func = nn.MSELoss()  # 均方损失函数

    def store_experiences(self, s, a, r, s_):  # 存储经验至buffer中
        transition = np.hstack((s, a, [r], s_))  # 转换为元组[s, a, r, s_]
        index = self.pointer % MEMORY_CAPACITY  # 装入经验，如果满了则替代掉最旧的经验
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, s):  # 选择动作
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.actor(s)[0].detach()  # 输出的同时切断反向传播

    def learn(self):
        # softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic.' + x + '.data)')
        # 从buffer中随机选出一个batch进行训练
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_trans = self.memory[indices, :]
        # extract data from mini-batch of transitions including s, a, r, s_
        batch_s = torch.FloatTensor(batch_trans[:, :self.s_dim])
        batch_a = torch.FloatTensor(batch_trans[:, self.s_dim:self.s_dim + self.a_dim])
        batch_r = torch.FloatTensor(batch_trans[:, -self.s_dim - 1: -self.s_dim])
        batch_s_ = torch.FloatTensor(batch_trans[:, -self.s_dim:])
        '''训练Actor'''
        # 输入s至actor，输出动作a，然后用critic计算Q(s,a)
        a = self.actor(batch_s)
        q = self.critic(batch_s, a)
        actor_loss = -torch.mean(q)  # 梯度上升q，替换为梯度下降-q
        self.actor_loss_list.append(actor_loss)
        # 更新actor的参数:梯度归零->计算梯度->反向传播
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        '''训练Critic'''
        # 利用r与s_计算出Q'(s_,a_),进而计算出Q(s,a)的target值
        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        q_target = batch_r + GAMMA * q_tmp
        # 计算Q(s,a)和loss
        q_eval = self.critic(batch_s, batch_a)
        td_error = self.loss_func(q_target, q_eval)
        self.critic_loss_list.append(td_error)
        # 更新critic的参数:梯度归零->计算梯度->反向传播
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

    def save_model(self):
        torch.save(self.actor.state_dict(), 'actor_weights.pth')
        torch.save(self.critic.state_dict(), 'critic_weights.pth')


# 配置gym
env = gym.make(ENV_NAME)
env = env.unwrapped
env.reset(seed=1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]  # 动作值为[a1,a2]，a1控制油门，a2控制左右点火引擎，取值范围都为[-1,1]的实数
a_bound = env.action_space.high
a_low_bound = env.action_space.low

ddpg = DDPG(a_dim, s_dim, a_bound)
var = 3  # 加入噪声用到的正态分布中的标准差
t1 = time.time()
reward_list = []
for i in range(EPISODES):
    s = env.reset()
    ep_r = 0  # 每一个episode的累积奖励值
    for j in range(EP_STEPS):
        if RENDER: env.render()
        # 加入噪声
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)
        s_, r, done, _, info = env.step(a)
        ddpg.store_experiences(s, a, r , s_)  # 存储与环境互动经验
        if ddpg.pointer > MEMORY_CAPACITY:
            var *= 0.9999  # decay the exploration controller factor
            ddpg.learn()

        s = s_
        ep_r += r
        if j == EP_STEPS - 1:
            reward_list.append(ep_r)
            print('Episode: ', i, ' Reward: %i' % ep_r, 'Explore: %.2f' % var)

    if i > 0 and i % 50 == 0:
        ddpg.save_model()
        x = range(0, i + 1)
        plt.plot(x, reward_list, '.-')
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.show()
print('Running time: ', time.time() - t1)
