import gym
import agents

EPISODES = 1000
MAX_STEPS = 1000

env = gym.make('FrozenLake8x8-v1')
env.render()

sarsa_agent = agents.SarsaAgent(env, 0.95, 0.95, 0.1)
q_agent = agents.QAgent(env, 0.95, 0.95, 0.1)
goal_count = 0

for episode in range(EPISODES):
    state = env.reset()
    step = 0
    done = False
    for step in range(MAX_STEPS):
        action = q_agent.choose_egreedy(state)
        new_state, reward, done, info = env.step(action)
        if reward == 1:
            goal_count += 1
        q_agent.update(state, action, reward, new_state)
        env.render()
        state = new_state
        if done:
            break
    
print(f'goal count: {goal_count}')
print(q_agent.Q)
