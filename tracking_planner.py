from typing import List, Tuple
from planner import Planner
import numpy as np
import klampt
from tracker import Tracker


class TrackingPlannerInstance(Planner):
    def __init__(self, world_model: klampt.WorldModel, dt: float):
        super().__init__(world_model, dt)

    def next(self) -> Tuple[List[float], List[float]]:
        pass


class TrackerPlanner:
    def __init__(self, tracker: Tracker, vel_set: np.ndarray=None):
        self.tracker = tracker
        if vel_set is None:
            vel_set = np.array([
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.5],
                [1.0, 0.0],
                [1.0, -0.5],
                [1.0, -1.0],
                [0.0, -1.0]
            ])
        self.vel_set = vel_set
