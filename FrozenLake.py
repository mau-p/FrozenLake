from xxlimited import new
import gym

from agents import SarsaAgent

EPISODES = 100
MAX_STEPS = 99

env = gym.make('FrozenLake8x8-v1')
env.render()

sarsa_agent = SarsaAgent(env, 0.95, 0.95, 0.1)

for episode in range(EPISODES):
    state = env.reset()
    step = 0
    done = False
    for step in range(MAX_STEPS):
        action = sarsa_agent.choose_egreedy(state)
        new_state, reward, done, info = env.step(action)
        next_action = sarsa_agent.choose_egreedy(new_state)
        sarsa_agent.update(state, new_state, reward, action, next_action)
        env.render()
        if done:
            break
