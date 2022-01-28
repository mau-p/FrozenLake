import gym
import agents
import matplotlib.pyplot as plt
import numpy as np

TRIALS = 500
EPISODES = 500
MAX_STEPS = 25


env = gym.make('FrozenLake-v1', is_slippery = False)

greedyAgentQ = agents.GreedyAgent(env, alpha=0.80, gamma=0.95, episodes=EPISODES)
egreedyAgentQ = agents.eGreedyAgent(env, epsilon=0.1, alpha=0.80, gamma=0.95, episodes=EPISODES)
greedyAgentSarsa = agents.GreedyAgent(env, alpha=0.8, gamma=0.95, episodes=EPISODES)
egreedyAgentSarsa = agents.eGreedyAgent(env, epsilon=0.1, alpha=0.80, gamma=0.95, episodes=EPISODES)

agent_list = []
agent_list.append(greedyAgentQ)
agent_list.append(egreedyAgentQ)
agent_list.append(greedyAgentSarsa)
agent_list.append(egreedyAgentSarsa)

average_stats = []
average_rewards = []

for agent in agent_list:
    print(f'Training agent {agent_list.index(agent)}...')
    for trial in range(TRIALS):
        for episode in range(EPISODES):
            state = env.reset()
            step = 0
            reached_goal = False
            done = False
            for step in range(MAX_STEPS): 
                action = agent.choose(state)
                new_state, reward, done, info = env.step(action)
                next_action = agent.choose(new_state)
                agent.sarsa_update(state, action, reward, new_state, next_action)
                if agent_list.index(agent) < 2:
                    agent.q_update(state, action, reward, new_state)
                else:
                    agent.sarsa_update(state, action, reward, new_state, next_action)
                state = new_state
                if done:
                    break 
            agent.update_stats(reward)
        agent.reset()
    average_stats.append(agent.prob_of_success / TRIALS)
    average_rewards.append(agent.obtained_rewards / TRIALS)


plt.plot(average_rewards[0], label="greedy q")
plt.plot(average_rewards[1], label="egreedy q")
plt.plot(average_rewards[2], label="greedy SARSA")
plt.plot(average_rewards[3], label="egreedy SARSA")

plt.legend(loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average reward")
plt.title(f'Performance of agents throughout learning')
plt.show()


