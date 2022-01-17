import numpy as np

class Agent:
    def __init__(self, env) -> None:
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n

    def init_q_values(self):
        return np.zeros((self.state_space, self.action_space))

    def choose_egreedy(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.Q[state:])

class SarsaAgent(Agent):
    def __init__(self, env, alpha, gamma, epsilon) -> None:
        Agent.__init__(self, env)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = self.init_q_values()
    
    def update(self, state, state2, reward, action, action2):
        predict = self.Q[state:]
        target = reward + self.gamma * self.Q[state2, action2]
        self.Q[state, action] += self.alpha * (target - predict)
    