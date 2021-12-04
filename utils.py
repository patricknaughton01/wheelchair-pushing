import json
from typing import List

import klampt
from klampt.model import ik
import numpy as np

from consts import SETTINGS_PATH


def extract_cfg(cfg: List[float], dofs: List[int]) -> List[float]:
    ret = []
    for d in dofs:
        ret.append(cfg[d])
    return ret


class WheelchairUtility:
    def __init__(self, model: klampt.RobotModel):
        self.model = model
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]
        self.base_name = "base_link"

    def cfg_to_rcfg(self, cfg: List[float]) -> np.ndarray:
        self.model.setConfig(cfg)
        tf = self.model.link(self.base_name).getTransform()
        yaw = np.arctan2(tf[0][1], tf[0][0])
        return np.array([tf[1][0], tf[1][1], yaw])

    def rcfg_to_cfg(self, rcfg: np.ndarray) -> List[float]:
        yaw = rcfg[2]
        c = np.cos(yaw)
        s = np.sin(yaw)
        r = (c,s,0,-s,c,0,0,0,1)
        goal = ik.objective(self.model.link(self.base_name), R=r,
            t=(*rcfg[:2], 0))
        if ik.solve_global(goal, activeDofs=self.wheelchair_dofs):
            return self.model.getConfig()
        print("COULDN'T SOLVE IK")
        print("rcfg: ", rcfg)
        print("R", r)
        return None
