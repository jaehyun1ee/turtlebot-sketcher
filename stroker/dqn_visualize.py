import gym
import numpy as np
import turtlebot3_env
import os, site

os.environ["LD_PRELOAD"] = site.USER_SITE + "/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0"

from stable_baselines3 import DQN

if __name__ == '__main__':
    # make environment
    env = gym.make('turtlebot3_env/Turtlebot3-real-v0')

    try:
        model = DQN("MlpPolicy", env, verbose=1)
        model = DQN.load("dqn_turtlebot")
        model.set_parameters("dqn_turtlebot")
        
        env.init_agent_origin(np.zeros(2) * 2 - 1)
        while True:
            rewards = []
            dists = []
            state = env.reset()
            pos_rand = np.zeros(2)
            while np.allclose(pos_rand, np.zeros(2)) or np.linalg.norm(pos_rand) > 1 or np.linalg.norm(pos_rand) < 0.02:
                pos_rand = np.random.rand(2) * 2 - 1
            env.set(pos_rand)
            while True:
                action, _ = model.predict(state, deterministic=True)
                past_state = state
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                dists.append(np.linalg.norm(state-past_state))
                #print(f"distance:{np.linalg.norm(state-past_state)}")
                if done:
                    print(f"current position: {state[0]:5f}, {state[1]:5f}, {state[2]:5f}")
                    print(f"target position: {state[5]:5f}, {state[6]:5f}, {state[7]:5f}")
                    print(f"reward: {sum(rewards)}")
                    print(f"dist: {np.mean(dists)}")
                    print(f"done: {done}")
                    #env.show()
                    state = env.reset()
                    break
            env.stop()
            k = int(input("ENTER 1 TO CONTINUE"))
            if k != 1:
                break
        input("WAIT TO CLOSE")
    finally:
        env.close()


