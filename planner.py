from typing import List, Tuple
import klampt
import numpy as np

class Planner:
    def __init__(self, world_model: klampt.WorldModel, dt: float):
        self.world_model = world_model
        self.dt = dt

    def plan(self, target: np.ndarray):
        """Perform any initial planning

        Args:
            target (np.ndarray): (3,) x,y,theta target in world frame
        """
        pass

    def next(self) -> Tuple[List[float], List[float]]:
        """Get the next configuration of TRINA and the wheelchair in the plan.

        Returns:
            Tuple[List[float], List[float]]: TRINA config, wheelchair config
        """
        raise NotImplementedError
