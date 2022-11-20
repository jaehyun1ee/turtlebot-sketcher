import gym
import turtlebot3_env
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    # make environment
    env = gym.make('turtlebot3_env/Turtlebot3-v0')
    try:
        conti = True
        while conti:
            # reset environment
            env.reset()

            # a random policy
            done = False
            its = 0
            rewards = 0
            while not done:
                action = 0
                next_state, reward, done, _ = env.step(action)
                print(f"it ({its});\nstate : {next_state}\naction : {action}\nreward : {reward}")
                #plt.imshow(np.transpose(env.grid, (1, 2, 0)))
                #plt.show()
                #plt.savefig('/grid.png')
                #input("PLT SAVED")

                its += 1
                if (its == 200):
                    break
                rewards += reward
                time.sleep(0.05)
            env.stop()
            print(f"final : {env.state}")
            print(f"tot_rew: {rewards}")

            # continue or not?
            key = int(input("ENTER 1 TO CONTINUE : "))
            if key != 1:
                conti = False

        # close environment
        input("ENTER TO CLOSE")
    finally:
        env.close()
