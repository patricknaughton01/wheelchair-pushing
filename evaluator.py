from typing import List, Tuple

import numpy as np
from consts import SETTINGS_PATH
from planner import Planner
import time
import json
import klampt
from klampt.math import so2, vectorops as vo
from tracking_planner import TrackingPlannerInstance
from utils import WheelchairUtility, extract_cfg


class Evaluator:
    def __init__(self, p: Planner, world_fn: str):
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.base_dofs: List[int] = self.settings["base_dofs"]
        self.wheelchair_dofs: List[int] = self.settings["wheelchair_dofs"]
        self.arm_dofs: List[int] = self.settings["left_arm_dofs"] + self.settings["right_arm_dofs"]
        self.p = p
        self.world_fn = world_fn
        self.world_model = klampt.WorldModel()
        self.world_model.loadFile(world_fn)
        self.robot_model: klampt.RobotModel = self.world_model.robot("trina")
        self.wheelchair_model: klampt.RobotModel = self.world_model.robot("wheelchair")
        self.wu = WheelchairUtility(self.wheelchair_model)
        self.target: np.ndarray = None
        self.disp_tol: float = 0
        self.rot_tol: float = 0
        self.stats = {}
        self.trajectory: List[Tuple[List[float], List[float]]] = []

    def eval(self, target: np.ndarray, disp_tol: float, rot_tol: float):
        self.eval_plan(target, disp_tol, rot_tol)
        self.eval_exec()
        self.eval_traj()

    def eval_plan(self, target: np.ndarray, disp_tol: float, rot_tol: float):
        self.target = target
        self.disp_tol = disp_tol
        self.rot_tol = rot_tol
        start = time.monotonic()
        self.p.plan(target, disp_tol, rot_tol)
        self.stats["planning_time"] = time.monotonic() - start

    def eval_exec(self):
        start_exec = time.monotonic()
        count = 0
        while True:
            try:
                cfgs = self.p.next()
                self.trajectory.append(cfgs)
                self.robot_model.setConfig(cfgs[0])
                self.wheelchair_model.setConfig(cfgs[1])
                count += 1
            except StopIteration:
                break
        wheelchair_cfg = self.wheelchair_model.getConfig()
        wheelchair_rcfg = self.wu.cfg_to_rcfg(wheelchair_cfg)
        wheelchair_xy = wheelchair_rcfg[:2]
        wheelchair_yaw = wheelchair_rcfg[2]
        if (np.linalg.norm(wheelchair_xy - self.target[:2]) <= self.disp_tol and
            abs(so2.diff(wheelchair_yaw, self.target[2])) <= self.rot_tol
        ):
            self.stats["success"] = True
        else:
            self.stats["success"] = False
        self.stats["execution_time"] = time.monotonic() - start_exec
        self.stats["execution_time"] += count * self.p.dt

    def eval_traj(self):
        trina_arm_cfg_dist = 0
        trina_base_t_dist = 0
        trina_base_r_dist = 0
        trina_strafe_dist = 0
        wheelchair_t_dist = 0
        wheelchair_r_dist = 0
        wheelchair_t_accel = 0
        wheelchair_r_accel = 0
        last_w_t_vel = None
        last_w_r_vel = None
        for i in range(len(self.trajectory) - 1):
            rc1 = self.trajectory[i][0]
            rc2 = self.trajectory[i+1][0]
            wc1 = self.trajectory[i][1]
            wc2 = self.trajectory[i+1][1]
            bc1 = extract_cfg(rc1, self.base_dofs)
            bc2 = extract_cfg(rc2, self.base_dofs)
            ac1 = extract_cfg(rc1, self.arm_dofs)
            ac2 = extract_cfg(rc2, self.arm_dofs)
            wbc1 = extract_cfg(wc1, self.wheelchair_dofs)
            wbc2 = extract_cfg(wc2, self.wheelchair_dofs)
            trina_arm_cfg_dist += vo.norm(vo.sub(ac1, ac2))
            trina_base_t_dist += vo.norm(vo.sub(bc1[:2], bc2[:2]))
            trina_base_r_dist += abs(so2.diff(bc1[2], bc2[2]))
            trina_strafe_dist += abs(bc1[1] - bc2[1])
            wheelchair_t_dist += vo.norm(vo.sub(wbc1[:2], wbc2[:2]))
            wheelchair_r_dist += abs(so2.diff(wbc1[2], wbc2[2]))
            w_t_vel = vo.div(vo.sub(wbc2[:2], wbc1[:2]), self.p.dt)
            w_r_vel = so2.diff(wbc2[2], wbc1[2]) / self.p.dt
            if i > 0:
                wheelchair_t_accel += vo.norm(vo.sub(w_t_vel, last_w_t_vel))**2
                wheelchair_r_accel += abs(w_r_vel - last_w_r_vel)**2
            last_w_t_vel = w_t_vel
            last_w_r_vel = w_r_vel
        n_a = len(self.trajectory) - 2
        self.stats["trina_arm_cfg_dist"] = trina_arm_cfg_dist
        self.stats["trina_base_t_dist"] = trina_base_t_dist
        self.stats["trina_base_r_dist"] = trina_base_r_dist
        self.stats["trina_strafe_dist"] = trina_strafe_dist
        self.stats["wheelchair_t_dist"] = wheelchair_t_dist
        self.stats["wheelchair_r_dist"] = wheelchair_r_dist
        self.stats["wheelchair_t_accel_rmsa"] = (wheelchair_t_accel / n_a)**0.5
        self.stats["wheelchair_r_accel_rmsa"] = (wheelchair_r_accel / n_a)**0.5


if __name__ == "__main__":
    world_fn = "Model/worlds/world_short_turn.xml"
    p = TrackingPlannerInstance(world_fn, 1 / 50)
    e = Evaluator(p, world_fn)
    e.eval(np.array([10.0, 0.0, 0.0]), 0.5, 0.5)
    print(e.stats)
