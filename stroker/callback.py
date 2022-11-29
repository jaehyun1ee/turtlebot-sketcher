from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self):
        dist = self.training_env.envs[0].dist_to_goal()
        self.logger.record("dist_to_goal", dist)
        return True
