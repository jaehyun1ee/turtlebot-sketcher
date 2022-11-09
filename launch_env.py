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
        for i in range(50):
            action = env.action_space.sample()
            next_state, _, _, _ = env.step(action)
            print("action : " + str(action))
            print("state : " + str(next_state))
        env.stop()
        print("final : " + str(env.state))

        # continue or not?
        key = int(input("ENTER 1 TO CONTINUE : "))
        if key != 1:
            conti = False

    # close environment
    input("ENTER TO CLOSE")
    env.close()
