from gym.envs.registration import register

register(
        id="turtlebot3_env/Turtlebot3-v0",
        entry_point="turtlebot3_env.envs:Turtlebot3GazeboEnv",
)
