import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm

# ランダム行動による攻略
class agent_random:
    def __init__(self, view=False):
        self.theta_0 = np.array([[np.nan, 1, 1, np.nan],
                                 [np.nan, 1, np.nan, 1],
                                 [np.nan, np.nan, 1, 1],
                                 [1, 1, 1, np.nan],
                                 [np.nan, np.nan, 1, 1],
                                 [1, np.nan, np.nan, np.nan],
                                 [1, np.nan, np.nan, np.nan],
                                 [1, 1, np.nan, np.nan]])
        self.pi_0 = self.simple_convert_into_pi_from_theta(self.theta_0)
        self.s = 0
        self.state_history=None
        if view:
            self.init_plt()
    def init_plt(self):
        self.fig = plt.figure(figsize=(5, 5))
        self.ax = plt.gca()
        plt.plot([1, 1], [0, 1], color='red', linewidth=2)
        plt.plot([1, 2], [2, 2], color='red', linewidth=2)
        plt.plot([2, 2], [2, 1], color='red', linewidth=2)
        plt.plot([2, 3], [1, 1], color='red', linewidth=2)

        plt.text(0.5, 2.5, 'S0', size=14, ha='center')
        plt.text(1.5, 2.5, 'S1', size=14, ha='center')
        plt.text(2.5, 2.5, 'S2', size=14, ha='center')
        plt.text(0.5, 1.5, 'S3', size=14, ha='center')
        plt.text(1.5, 1.5, 'S4', size=14, ha='center')
        plt.text(2.5, 1.5, 'S5', size=14, ha='center')
        plt.text(0.5, 0.5, 'S6', size=14, ha='center')
        plt.text(1.5, 0.5, 'S7', size=14, ha='center')
        plt.text(2.5, 0.5, 'S8', size=14, ha='center')
        plt.text(0.5, 2.3, 'START', ha='center')
        plt.text(2.5, 0.3, 'GOAL', ha='center')

        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 3)
        plt.tick_params(axis='both', which='both', bottom='off', top='off',
                        labelbottom='off', right='off', left='off', labelleft='off')
        self.line, = self.ax.plot([0.5], [2.5], marker='o', color='g', markersize=60)
    def simple_convert_into_pi_from_theta(self, theta):
        pi = np.zeros(theta.shape)
        for i in range(len(pi)):
            pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
        pi = np.nan_to_num(pi)
        return pi
    def get_next_s(self, pi, s):
        direction = ['up','right','down','left']

        next_direction = np.random.choice(direction, p=pi[s, :])
        if next_direction=='up':
            s_next = s - 3
        elif next_direction=='right':
            s_next = s + 1
        elif next_direction=='down':
            s_next = s + 3
        elif next_direction=='left':
            s_next = s - 1

        return s_next
    def goal_maze(self):
        state_history = [0]

        while(1):
            next_s = self.get_next_s(self.pi_0, self.s)
            state_history.append(next_s)
            if next_s==8:
                break
            else:
                self.s = next_s
        self.state_history = state_history
        return state_history
    def draw_init(self):
        self.line.set_data([], [])
        return (self.line,)
    def draw_maze(self,i):
        pos = self.state_history[i]
        x = (pos % 3) + 0.5
        y = 2.5 - int(pos / 3)
        self.line.set_data(x, y)
        return (self.line,)
    def draw_anim(self,):
        anim = animation.FuncAnimation(self.fig, self.draw_maze, init_func=self.draw_init, frames=len(self.state_history), interval=200,
                                       repeat=False)
        anim.save('out.html', writer='imagemagick')

# 方策勾配法による攻略
class agent_policy(agent_random):
    def __init__(self):
        self.theta_0 = np.array([[np.nan, 1, 1, np.nan],
                                 [np.nan, 1, np.nan, 1],
                                 [np.nan, np.nan, 1, 1],
                                 [1, 1, 1, np.nan],
                                 [np.nan, np.nan, 1, 1],
                                 [1, np.nan, np.nan, np.nan],
                                 [1, np.nan, np.nan, np.nan],
                                 [1, 1, np.nan, np.nan]])
        self.theta = self.theta_0.copy()
        self.softmax_convert_into_pi_from_theta()
    def softmax_convert_into_pi_from_theta(self):
        beta = 1.0
        pi = np.zeros(self.theta.shape)
        exp_theta = np.exp(beta * self.theta)
        for i in range(len(pi)):
            pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
        self.pi = np.nan_to_num(pi)
    def get_next_s(self):
        direction = ['up','right','down','left']
        action=0
        next_direction = np.random.choice(direction, p=self.pi[self.s, :])
        if next_direction=='up':
            action=0
            s_next = self.s - 3
        elif next_direction=='right':
            action=1
            s_next = self.s + 1
        elif next_direction=='down':
            action=2
            s_next = self.s + 3
        elif next_direction=='left':
            action=3
            s_next = self.s - 1

        return [action, s_next]
    def goal_maze(self, train=True):
        if train:
            self.train_maze()
        state_history = [[0, np.nan]]
        self.s = 0
        while(1):
            [action, next_s] = self.get_next_s()
            state_history[-1][1] = action
            state_history.append([next_s, np.nan])
            if next_s==8:
                break
            else:
                self.s = next_s
        self.state_history = np.array(state_history)[:,0]
        return state_history
    def update_theta(self, s_a_history):
        eta = 0.1
        T = len(s_a_history) - 1
        [m, n] = self.theta.shape
        delta_theta = self.theta.copy()
        for i in range(m):
            for j in range(n):
                if not(np.isnan(self.theta[i][j])):
                    SA_i = [SA for SA in s_a_history if SA[0] == i]
                    SA_ij = [SA for SA in s_a_history if SA == [i,j]]
                    N_i = len(SA_i)
                    N_ij = len(SA_ij)
                    delta_theta[i, j] = (N_ij + self.pi[i, j] * N_i) / T
        self.theta = self.theta + eta * delta_theta
        return delta_theta
    def train_maze(self):
        count=0
        while(1):
            count+=1
            hist = self.goal_maze(False)
            self.update_theta(hist)
            delta_pi = self.pi
            self.softmax_convert_into_pi_from_theta()
            if np.sum(np.abs(self.pi - delta_pi)) < 10e-8:
                print(count, '回で終了')
                break

# Sarsaによる攻略
class agent_sarsa(agent_policy):
    def __init__(self):
        self.theta_0 = np.array([[np.nan, 1, 1, np.nan],
                                 [np.nan, 1, np.nan, 1],
                                 [np.nan, np.nan, 1, 1],
                                 [1, 1, 1, np.nan],
                                 [np.nan, np.nan, 1, 1],
                                 [1, np.nan, np.nan, np.nan],
                                 [1, np.nan, np.nan, np.nan],
                                 [1, 1, np.nan, np.nan]])
        self.theta = self.theta_0.copy()
        self.softmax_convert_into_pi_from_theta()
        self.s = 0

        #parameter
        self.epsilon = 0.5
        self.eta = 0.1
        self.gamma = 0.9
        self.Q = np.random.rand(len(self.theta), len(self.theta[0])) * self.theta
    def get_action(self,s):
        direction = ['up','right','down','left']
        if np.random.rand()<self.epsilon:
            next_direction = np.random.choice(direction, p=self.pi[s,:])
        else:
            next_direction = direction[np.nanargmax(self.Q[s, :])]
        if next_direction=='up':
            action=0
        elif next_direction=='right':
            action=1
        elif next_direction=='down':
            action=2
        elif next_direction=='left':
            action=3
        return action
    def get_next_s(self, action):
        if action==0:
            s_next = self.s - 3
        elif action==1:
            s_next = self.s + 1
        elif action==2:
            s_next = self.s + 3
        elif action==3:
            s_next = self.s - 1
        return s_next
    def sarsa(self, s, a, r, s_next, a_next, ):
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r + self.gamma * self.Q[s_next, a_next] - self.Q[s, a])
    def goal_maze(self, train=True):
        if train:
            self.train_maze()
        state_history = [[0, np.nan]]
        self.s = 0
        a_next = self.get_action(self.s)
        while(1):
            a = a_next
            state_history[-1][1] = a
            s_next = self.get_next_s(a_next)
            state_history.append([s_next, np.nan])
            if s_next==8:
                r = 1
                a_next = np.nan
            else:
                r = 0
                a_next = self.get_action(s_next)
            self.sarsa(self.s, a, r, s_next, a_next)

            if s_next == 8:
                break
            else:
                self.s = s_next
        self.state_history = np.array(state_history)[:,0]
        return state_history
    def train_maze(self):
        count=0
        for i in range(100):
            hist = self.goal_maze(False)

# Q学習による攻略
class agent_Q(agent_policy):
    def __init__(self):
        self.theta_0 = np.array([[np.nan, 1, 1, np.nan],
                                 [np.nan, 1, np.nan, 1],
                                 [np.nan, np.nan, 1, 1],
                                 [1, 1, 1, np.nan],
                                 [np.nan, np.nan, 1, 1],
                                 [1, np.nan, np.nan, np.nan],
                                 [1, np.nan, np.nan, np.nan],
                                 [1, 1, np.nan, np.nan]])
        self.theta = self.theta_0.copy()
        self.softmax_convert_into_pi_from_theta()
        self.s = 0

        #parameter
        self.epsilon = 0.5
        self.eta = 0.1
        self.gamma = 0.9
        self.Q = np.random.rand(len(self.theta), len(self.theta[0])) * self.theta
        self.V = []
    def get_action(self,s):
        direction = ['up','right','down','left']
        if np.random.rand()<self.epsilon:
            next_direction = np.random.choice(direction, p=self.pi[s,:])
        else:
            next_direction = direction[np.nanargmax(self.Q[s, :])]
        if next_direction=='up':
            action=0
        elif next_direction=='right':
            action=1
        elif next_direction=='down':
            action=2
        elif next_direction=='left':
            action=3
        return action
    def get_next_s(self, action):
        if action==0:
            s_next = self.s - 3
        elif action==1:
            s_next = self.s + 1
        elif action==2:
            s_next = self.s + 3
        elif action==3:
            s_next = self.s - 1
        return s_next
    def Q_learn(self, s, a, r, s_next):
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r + self.gamma * np.nanmax(self.Q[s_next, :]) - self.Q[s, a])
    def goal_maze(self, train=True):
        if train:
            self.train_maze()
        state_history = [[0, np.nan]]
        self.s = 0
        while(1):
            a = self.get_action(self.s)
            state_history[-1][1] = a
            s_next = self.get_next_s(a)
            state_history.append([s_next, np.nan])
            if s_next==8:
                r = 1
                a_next = np.nan
            else:
                r = 0
            self.Q_learn(self.s, a, r, s_next)

            if s_next == 8:
                break
            else:
                self.s = s_next
        self.state_history = np.array(state_history)[:,0]
        return state_history
    def train_maze(self):
        count=0
        for i in range(100):
            hist = self.goal_maze(False)
            self.V.append(np.nanmax(self.Q, axis=1))
            self.epsilon *= 0.8
    def init_plt(self):
        self.fig = plt.figure(figsize=(5, 5))
        self.ax = plt.gca()
        plt.plot([1, 1], [0, 1], color='red', linewidth=2)
        plt.plot([1, 2], [2, 2], color='red', linewidth=2)
        plt.plot([2, 2], [2, 1], color='red', linewidth=2)
        plt.plot([2, 3], [1, 1], color='red', linewidth=2)

        plt.text(0.5, 2.5, 'S0', size=14, ha='center')
        plt.text(1.5, 2.5, 'S1', size=14, ha='center')
        plt.text(2.5, 2.5, 'S2', size=14, ha='center')
        plt.text(0.5, 1.5, 'S3', size=14, ha='center')
        plt.text(1.5, 1.5, 'S4', size=14, ha='center')
        plt.text(2.5, 1.5, 'S5', size=14, ha='center')
        plt.text(0.5, 0.5, 'S6', size=14, ha='center')
        plt.text(1.5, 0.5, 'S7', size=14, ha='center')
        plt.text(2.5, 0.5, 'S8', size=14, ha='center')
        plt.text(0.5, 2.3, 'START', ha='center')
        plt.text(2.5, 0.3, 'GOAL', ha='center')

        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 3)
        plt.tick_params(axis='both', which='both', bottom='off', top='off',
                        labelbottom='off', right='off', left='off', labelleft='off')
    def draw_init(self):
        self.ax = plt.gca()
    def draw_maze(self,i):
        line, = self.ax.plot([0.5],[2.5],marker='s',color=cm.jet(self.V[i][0]), markersize=85)
        line, = self.ax.plot([1.5], [2.5], marker='s', color=cm.jet(self.V[i][1]), markersize=85)
        line, = self.ax.plot([2.5], [2.5], marker='s', color=cm.jet(self.V[i][2]), markersize=85)
        line, = self.ax.plot([0.5], [1.5], marker='s', color=cm.jet(self.V[i][3]), markersize=85)
        line, = self.ax.plot([1.5], [1.5], marker='s', color=cm.jet(self.V[i][4]), markersize=85)
        line, = self.ax.plot([2.5], [1.5], marker='s', color=cm.jet(self.V[i][5]), markersize=85)
        line, = self.ax.plot([0.5], [0.5], marker='s', color=cm.jet(self.V[i][6]), markersize=85)
        line, = self.ax.plot([1.5], [0.5], marker='s', color=cm.jet(self.V[i][7]), markersize=85)
        line, = self.ax.plot([2.5], [0.5], marker='s', color=cm.jet(1.0), markersize=85)
        return (line,)
    def draw_anim(self):
        anim = animation.FuncAnimation(self.fig, self.draw_maze, init_func=self.draw_init, frames=len(self.V), interval=200,
                                       repeat=False)
        anim.save('out.html', writer='imagemagick')

# 1000回試行の平均
def evaluate(model, num=1000):
    results=[]
    for _ in range(num):
        a = model()
        result = a.goal_maze()
        results.append(len(result))
    plt.hist(results, bins=20)
    plt.show()
    print(np.mean(results))

if __name__=='__main__':
    evaluate(agent_Q)