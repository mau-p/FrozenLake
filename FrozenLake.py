import gym

from agents import SarsaAgent

ITERATIONS = 20

env = gym.make('FrozenLake8x8-v1')
env.reset()
env.render()

sarsa_agent = SarsaAgent(env, 0.95, 0.95, 0.1)

for i in range(ITERATIONS):
    state = env.reset()
    action = sarsa_agent.choose_egreedy(state)
    new_state, reward, done, info = env.step(action)
    next_action = sarsa_agent.choose_egreedy(new_state)
    sarsa_agent.update(state, new_state, reward, action, next_action)
    env.render()
    if done:
        break
