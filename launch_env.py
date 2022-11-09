import gym
import turtlebot3_env

if __name__ == '__main__':
    # make environment
    env = gym.make('turtlebot3_env/Turtlebot3-v0')

    conti = True
    while conti:
        # reset environment
        env.reset()

        # a random policy
        done = False
        its = 0
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            print("it (" + str(its) + ") ; " + "\nstate : " + str(next_state) + "\naction : " + str(action) + "\nreward : " + str(reward))

            its += 1
            if (its == 200):
                break
        env.stop()
        print("final : " + str(env.state))

        # continue or not?
        key = int(input("ENTER 1 TO CONTINUE : "))
        if key != 1:
            conti = False

    # close environment
    input("ENTER TO CLOSE")
    env.close()
