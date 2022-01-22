import gym
import agents
import matplotlib.pyplot as plt


EPISODES = 1000
MAX_STEPS = 200


env = gym.make('FrozenLake-v1', is_slippery = False)
env.render()
#agent = agents.GreedyAgent(env, alpha=0.95, gamma=0.95, steps=MAX_STEPS, episodes=EPISODES)
#agent = agents.eGreedyAgent(env, epsilon=0.1, alpha=0.95, gamma=0.95,steps=MAX_STEPS,episodes=EPISODES)
agent = agents.OptimisticAgent(env, alpha=0.95, gamma=0.95,steps=MAX_STEPS,episodes=EPISODES)
#agent = agents.UCB(env, alpha=0.95, gamma=0.95,steps=MAX_STEPS,episodes=EPISODES)

for episode in range(EPISODES):
    state = env.reset()
    step = 0
    reached_goal = False
    done = False
    for step in range(MAX_STEPS):
        action = agent.choose(state)
        new_state, reward, done, info = env.step(action)
        next_action = agent.choose(new_state)
        env.render()
        agent.sarsa_update(state, action, reward, new_state, next_action)
        #agent.q_update(state, action, reward, new_state)
        state = new_state
        if done:
            break 
    agent.update_stats(reward)

plt.plot(agent.obtained_rewards, label="Amount of time it reaches the goal")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Number of optimal actions in %")
plt.title(f'Number of successful runs in {EPISODES} trainings')
plt.show()

plt.plot(agent.prob_of_success, label="xxx agent using xxx ")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Probability of the agent reaching the goal")
plt.title(f"Probability of the agent reaching the goal throughout {EPISODES} episodes")
plt.show()
print(agent.Q)

