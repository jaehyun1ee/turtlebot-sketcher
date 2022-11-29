from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self):
        dist = self.training_env.envs[0].dist_to_goal() * 64
        self.logger.record("pixel_difference", dist)
        return True
