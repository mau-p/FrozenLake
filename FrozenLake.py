import gym
import agents
import matplotlib.pyplot as plt


EPISODES = 1000
MAX_STEPS = 20


env = gym.make('FrozenLake-v1', is_slippery = False)
env.render()
greedy_agent = agents.GreedyAgent(env, alpha=0.95, gamma=0.95, steps=MAX_STEPS, episodes=EPISODES)

for episode in range(EPISODES):
    state = env.reset()
    step = 0
    reached_goal = False
    done = False
    for step in range(MAX_STEPS):
        action = greedy_agent.choose(state)
        new_state, reward, done, info = env.step(action)
        next_action = greedy_agent.choose(new_state)
        env.render()
        greedy_agent.sarsa_update(state, action, reward, new_state, next_action)
        state = new_state
        if done:
            break 
    greedy_agent.update_stats(reward)

plt.plot(greedy_agent.obtained_rewards, label="Amount of time it reaches the goal")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Number of optimal actions in %")
plt.title(f'Number of successful runs in {EPISODES} trainings')
plt.show()

plt.plot(greedy_agent.prob_of_success, label="Amount of time it reaches the goal")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Probability of the agent reaching the goal")
plt.title("Probability of the agent reaching the goal throughout training")
plt.show()


