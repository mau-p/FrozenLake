import gym
import agents
import matplotlib.pyplot as plt
import numpy as np

TRIALS = 50
EPISODES = 1000
MAX_STEPS = 25

alphas = [0.75, 0.8, 0.85, 0.90, 0.95, 1]
gammas = [0.75, 0.8, 0.85, 0.90, 0.95, 1]

env = gym.make('FrozenLake-v1', is_slippery = False)
average_stats = []
max_reward = 0
best_combo = (0,0)

for gamma in gammas:
    for alpha in alphas:
        print(f'trying combo {gamma} {alpha}')
        agent = agents.GreedyAgent(env, alpha, gamma, episodes=EPISODES)
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
                    # env.render()
                    # if agent_list.index(agent) < 2:
                    #     agent.q_update(state, action, reward, new_state)
                    # else:
                    #     agent.sarsa_update(state, action, reward, new_state, next_action)
                    agent.q_update(state, action, reward, new_state)
                    state = new_state
                    if done:
                        break 
                agent.update_stats(reward)
            agent.reset()
        average_stats.append(agent.prob_of_success / TRIALS)
        result_array = average_stats[0]
        last_reward = result_array[-1]
        if last_reward > max_reward:
            max_reward - last_reward
            best_combo = (alpha, gamma)

print(f'best comba {best_combo}')