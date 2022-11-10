import gym
import numpy as np
import turtlebot3_env

env = gym.make('turtlebot3_env/Turtlebot3Discrete-v0')
q_table = np.zeros((5, 5, 5, 3))
buckets = np.array([5, 5, 5])
alpha = 0.3
gamma = 0.99
rewards = []

def to_discrete_states(state):
    interval = np.zeros(len(state["agent"]), dtype=int)
    max_range = np.array([3, 3, 3])

    inter = np.floor((np.array(state["agent"]) + max_range)/(2*max_range/buckets))
    
    for i in range(len(inter)):
        if inter[i] >= buckets[i]: interval[i] = buckets[i] - 1
        elif inter[i] < 0: interval[i] = 0
        else: interval[i] = int(inter[i])

    return interval

def get_action(state, t):
    if np.random.random() < max(0.001, min(0.015, 1.0 - np.log10((t+1)/220))):
        return env.action_space.sample()
    interval = to_discrete_states(state)

    return np.argmax(np.array(q_table[tuple(interval)]))

def update_q(state, reward, action, next_state, t):
    interval = to_discrete_states(state)
    next_interval = to_discrete_states(next_state)
    q_next = max(q_table[tuple(next_interval)])


    q_table[tuple(interval)][action]+= max(0.4, min(0.1, 1.0 - np.log10((t+1)/125))) * (reward + gamma*q_next - q_table[tuple(interval)][action])

if __name__ == '__main__':
    for episode in range(1000):
        state, _ = env.reset()
        t = 0
        while True:
            action = get_action(state, episode)
            next_state, reward, done, _ = env.step(action)
            update_q(state, reward, action, next_state, t)
            state = next_state
            print(state, reward, action)
            t+=1
            if done:
                print("Episode finishied")
                rewards.append(reward)
                break
            if t > 500:
                break
    print(rewards)
    env.close()
