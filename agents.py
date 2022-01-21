import numpy as np


class Agent:
    def __init__(self, env, alpha, gamma) -> None:
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.Q = self.init_q_values()
        self.action_count = [0] * env.action_space.n
        self.alpha = alpha
        self.gamma = gamma
        

    def init_q_values(self):
        return np.zeros((self.state_space, self.action_space))

    def sarsa_update(self, state, state2, reward, action, action2):
        predict = self.Q[(state, action)]
        target = reward + self.gamma * self.Q[(state2, action2)]
        self.Q[(state, action)] += self.alpha * (target - predict)

    def q_update(self, state, action, reward, new_state):
        self.Q[(state, action)] += self.alpha * (reward + self.gamma * np.argmax(self.Q[new_state, :]) - self.Q[state, action])

class GreedyAgent(Agent):
    def __init__(self, env, alpha, gamma) -> None:
        super().__init__(env, alpha, gamma)

    def choose(self, state):
        return np.random.choice(np.where(self.Q[state,:] == self.Q[state,:].max())[0]) # returns random argmax

class eGreedyAgent(Agent):
    def __init__(self, env, epsilon, alpha, gamma) -> None:
        super().__init__(env, alpha, gamma)
        self.epsilon = epsilon
    
    def choose(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.random.choice(np.where(self.Q[state,:] == self.Q[state,:].max())[0]) # returns random argmax

class OptimisticAgent(Agent):
    def __init__(self, env, alpha, gamma) -> None:
        super().__init__(env, alpha, gamma)
        self.Q = np.ones((self.state_space, self.action_space))

    def choose(self, state):
        return np.random.choice(np.where(self.Q[state,:] == self.Q[state,:].max())[0]) # returns random argmax


class UCB(Agent):
    def __init__(self, env, c, alpha, gamma) -> None:
        super().__init__(env, alpha, gamma)
        self.c = c
        self.t = 1

    def decision(self, state):
        max_combined = -10
        for action in enumerate(self.Q[state,:]):
            action_counter = self.action_count[action] + 1
            uncertainty = self.c * np.sqrt(np.log(self.t)/action_counter)
            if uncertainty + self.Q[(state, action)] > max_combined:
                max_combined = uncertainty + self.Q(state, action)
                max_combined_action = action
        self.t += 1

        return max_combined_action

