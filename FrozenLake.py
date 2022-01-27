import gym
import agents
import matplotlib.pyplot as plt
import numpy as np

TRIALS = 100
EPISODES = 1000
MAX_STEPS = 25


env = gym.make('FrozenLake-v1', is_slippery = False)
greedyAgentQ = agents.GreedyAgent(env, alpha=0.80, gamma=0.85, episodes=EPISODES)
egreedyAgentQ = agents.eGreedyAgent(env, epsilon=0.1, alpha=0.80, gamma=0.85, episodes=EPISODES)
greedyAgentSarsa = agents.GreedyAgent(env, alpha=0.85, gamma=0.85, episodes=EPISODES)
egreedyAgentSarsa = agents.eGreedyAgent(env, epsilon=0.1, alpha=0.80, gamma=0.85, episodes=EPISODES)

agent_list = []
agent_list.append(greedyAgentQ)
agent_list.append(egreedyAgentQ)
agent_list.append(greedyAgentSarsa)
agent_list.append(egreedyAgentSarsa)

average_stats = []

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


result_array = average_stats[0]
print(result_array[-1])

plt.plot(average_stats[0], label="greedy Q")
plt.plot(average_stats[1], label="egreedy Q")
plt.plot(average_stats[2], label="greedy Sarsa")
plt.plot(average_stats[3], label="egreedy Sarsa")
plt.legend(loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average reward per trial%")
plt.title(f'Number of successful runs in {EPISODES} trainings')
plt.show()


