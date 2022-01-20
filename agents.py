import numpy as np


class Agent:
    def __init__(self, env) -> None:
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.Q = self.init_q_values()

    def choose_egreedy(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.random.choice(np.where(self.Q[state,:] == self.Q[state,:].max())[0]) # returns random argmax

    def init_q_values(self):
        return np.zeros((self.state_space, self.action_space))

class SarsaAgent(Agent):
    def __init__(self, env, alpha, gamma, epsilon) -> None:
        Agent.__init__(self, env)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def update(self, state, state2, reward, action, action2):
        predict = self.Q[state, action]
        target = reward + self.gamma * self.Q[state2, action2]
        self.Q[state, action] += self.alpha * (target - predict)


class QAgent(Agent):
    def __init__(self, env, alpha, gamma, epsilon) -> None:
        super().__init__(env)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def update(self, state, action, reward, new_state):
        self.Q[state, action] += self.alpha * (reward + self.gamma * np.argmax(self.Q[new_state, :]) - self.Q[state, action])
