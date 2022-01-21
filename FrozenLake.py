import gym
import agents
import matplotlib.pyplot as plt
import numpy as np


EPISODES = 100
MAX_STEPS = 100


env = gym.make('FrozenLake-v1')
env.render()
greedy_agent = agents.GreedyAgent(env, alpha=0.95, gamma=0.95)
goal_count = np.zeros(EPISODES)
count = 0
agent_succes = np.zeros(EPISODES)

for episode in range(EPISODES):
    state = env.reset()
    step = 0
    done = False
    for step in range(MAX_STEPS):
        action = greedy_agent.choose(state)
        new_state, reward, done, info = env.step(action)
        if reward == 1:
            goal_count += 1
        greedy_agent.q_update(state, action, reward, new_state)
        env.render()
        state = new_state
        if done:
            break
            
    agent_succes[training] = agent_succes[training] + (count - agent_succes[training]) / (training + 1)

    goal_count = np.zeros(EPISODES)
    count=0

plt.plot(agent_succes, label="Amount of time it reaches the goal")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Number of optimal actions in %")
plt.title("Number of successful runs in " + str(EPISODES) + " trainings")
plt.show()
#print(q_agent.Q)
print(agent_succes)


