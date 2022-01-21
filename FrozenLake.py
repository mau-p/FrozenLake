import gym
import agents

EPISODES = 10
MAX_STEPS = 20

env = gym.make('FrozenLake-v1')
env.render()

greedy_agent = agents.GreedyAgent(env, alpha=0.95, gamma=0.95)
goal_count = 0

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

