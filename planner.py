from typing import List, Tuple
import klampt
import numpy as np


class Planner:
    def __init__(self, world_model: klampt.WorldModel, dt: float):
        self.world_model = world_model
        self.dt = dt
        self.target: np.ndarray = None
        self.disp_tol: float = None
        self.rot_tol: float = None

    def plan(self, target: np.ndarray, disp_tol: float, rot_tol: float):
        """Perform any initial planning

        Args:
            target (np.ndarray): (3,) x,y,theta target in world frame
        """
        self.target = target
        self.disp_tol = disp_tol
        self.rot_tol = rot_tol

    def next(self) -> Tuple[List[float], List[float]]:
        """Get the next configuration of TRINA and the wheelchair in the plan.

        Raises:
            StopIteration: Raised when execution is finished.

        Returns:
            Tuple[List[float], List[float]]: TRINA config, wheelchair config
        """
        raise NotImplementedError

    def _check_target(self):
        """Checks that the target has been set. Raises error if not, otherwise
        simply returns.

        Raises:
            ValueError: Raised if target has not been set.
        """
        if self.target is None:
            raise ValueError("Target is None")
