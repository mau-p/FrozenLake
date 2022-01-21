import numpy as np


class Agent:
    def __init__(self, env, alpha, gamma, steps, episodes) -> None:
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.Q = self.init_q_values() # Set up Q table
        self.action_count = [0] * env.action_space.n # Counts the amount of times the agent has taken each action
        self.obtained_rewards = np.zeros(episodes) # Array that keeps track of the obtained reward per episode
        self.prob_of_success = np.zeros(episodes) # Probability of succes at each iteration
        self.total_success = 0 # counter of how many successes the agent has had
        self.alpha = alpha # alpha parameter for Q and Sarsa Learning
        self.gamma = gamma # Gamma parameter for Q and Sarsa Learning
        self.index = 0      # Internal tracking of which episode the agent is at. 
 

    def init_q_values(self):
        return np.zeros((self.state_space, self.action_space))

    def sarsa_update(self, state, action, reward, state2, action2):
        predict = self.Q[(state, action)]
        target = reward + self.gamma * self.Q[(state2, action2)]
        self.Q[(state, action)] += self.alpha * (target - predict)

    def q_update(self, state, action, reward, new_state):
        self.Q[(state, action)] += self.alpha * (reward + self.gamma * np.argmax(self.Q[new_state, :]) - self.Q[state, action])

    def update_stats(self, reward):
        self.obtained_rewards[self.index] = reward
        if reward == 1:
            self.total_success += 1
        self.index += 1
        self.prob_of_success[self.index-1] = self.total_success/self.index

        
class GreedyAgent(Agent):
    def __init__(self, env, alpha, gamma, steps, episodes) -> None:
        super().__init__(env, alpha, gamma, steps, episodes)

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

