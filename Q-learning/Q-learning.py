import numpy as np
import pandas as pd
import time


N_STATES = 6  # 状态数量
ACTIONS = ['left', 'right']  # available actions
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # 折扣因子
MAX_EPISODES = 13  # maximum episodes
FRESH_TIME = 0.3  # 移动一次后的刷新时间


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table 初始化全为0
        columns=actions,
    )
    return table


def choose_action(state, q_table):
    # epsilon-greedy策略
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # 随机选择动作
        action_name = np.random.choice(ACTIONS)
    else:  # 贪婪策略
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(S, A):
    # 在状态S下采取动作A，返回下一个状态S_和环境的奖励R
    if A == 'right':  # move right
        if S == N_STATES - 2:  # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # 更新环境
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T'
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # 强化学习算法部分
    Qtable = build_q_table(N_STATES, ACTIONS)
    for epoch in range(MAX_EPISODES):
        step = 0
        S = 0
        is_terminal = False
        while not is_terminal:
            A = choose_action(S, Qtable)
            step += 1
            S_, R = get_env_feedback(S, A)
            update_env(S_, epoch, step)
            Q_predict = Qtable.iloc[S][A]
            if S_ == 'terminal':
                Q_real = R
                is_terminal = True
            else:
                Q_real = R + GAMMA * Qtable.iloc[S_, :].max()
            Qtable.iloc[S][A] += ALPHA * (Q_real - Q_predict)
            S = S_
    return Qtable


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
