import numpy as np
from tracker import Tracker

class Planner:
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
