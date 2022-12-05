import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.n_episodes = 0

    def _on_step(self):
        done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        if np.sum(done_array) is None:
            return True
        self.n_episodes += np.sum(done_array).item()
        if np.sum(done_array).item() > 0:
            print(self.n_episodes)
            dist = self.training_env.envs[0].dist_to_goal() * 64
            self.logger.record_mean("pixel_difference", dist)

        return True
