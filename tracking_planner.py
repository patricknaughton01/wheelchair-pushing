import json
from typing import List, Tuple

from klampt.math import so2
from consts import SETTINGS_PATH
from planner import Planner
import numpy as np
import klampt
from grid_planner import GridPlanner
from tracker import Tracker
from utils import WheelchairUtility


class TrackingPlannerInstance(Planner):
    def __init__(self, world_fn: str, dt: float):
        super().__init__(world_fn, dt)
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.tracker_weights = {
            "arm_penalty": 0,
            "strafe_penalty": 1,
            "base_penalty": 0,
            "attractor_penalty": 10
        }
        self.gp_res = 0.5
        self.rollout = 20
        self.robot_model: klampt.RobotModel = self.world_model.robot("trina")
        self.wheelchair_model: klampt.RobotModel = self.world_model.robot("wheelchair")
        self.wu = WheelchairUtility(self.wheelchair_model)
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]
        self.tracker = Tracker(self.world_model, self.dt, self.tracker_weights)
        self.executor = None
        self.cfgs_buffer: List[Tuple[List[float], List[float]]] = []
        self.cfg_ind = 0

    def plan(self, target: np.ndarray, disp_tol: float, rot_tol: float):
        super().plan(target, disp_tol, rot_tol)
        cfg = self.tracker.wheelchair_model.getConfig()
        t_cfg = self.wu.rcfg_to_cfg(target)
        self.tracker.wheelchair_model.setConfig(cfg)
        gp = GridPlanner(self.world_fn, t_cfg, self.gp_res)
        self.executor = TrackerExecutor(self.tracker, cfg, gp, self.disp_tol,
            self.rot_tol, rollout=self.rollout)
        # warm start the grid planner
        gp.get_dist(self.wu.cfg_to_rcfg(
            self.wheelchair_model.getConfig()))

    def next(self):
        self._check_target()
        # Check for termination:
        wheelchair_cfg = self.wheelchair_model.getConfig()
        wheelchair_rcfg = self.wu.cfg_to_rcfg(wheelchair_cfg)
        wheelchair_xy = wheelchair_rcfg[:2]
        wheelchair_yaw = wheelchair_rcfg[2]
        if np.linalg.norm(wheelchair_xy - self.target[:2]) <= self.disp_tol:
            if abs(so2.diff(wheelchair_yaw, self.target[2])) <= self.rot_tol:
                raise StopIteration
        if self.cfg_ind >= len(self.cfgs_buffer):
            self.cfgs_buffer = self.executor.get_next_configs()
            self.cfg_ind = 0
        if self.cfg_ind is None:
            raise StopIteration
        self.robot_model.setConfig(self.cfgs_buffer[self.cfg_ind][0])
        self.wheelchair_model.setConfig(self.cfgs_buffer[self.cfg_ind][1])
        ret_ind = self.cfg_ind
        self.cfg_ind += 1
        return self.cfgs_buffer[ret_ind]


class TrackerExecutor:
    def __init__(self, tracker: Tracker, target_cfg: List[float],
        gp: GridPlanner, disp_tol: float, rot_tol: float,
        rollout: int=4, vel_set: np.ndarray=None
    ):
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.tracker = tracker
        self.target_cfg = target_cfg
        self.disp_tol = disp_tol,
        self.rot_tol = rot_tol
        self.rollout = rollout
        if vel_set is None:
            vel_set_list = []
            for fv in [0.0, 0.1, 1.0]:
                for rv in [0.0, 0.3, 1.0]:
                    if not (fv < 1e-3 and rv < 1e-3):
                        vel_set_list.append([fv, rv])
                        vel_set_list.append([fv, -rv])
            vel_set = np.array(vel_set_list)
        self.vel_set = vel_set
        self.wu = WheelchairUtility(self.tracker.wheelchair_model)
        self.target_np = self.wu.cfg_to_rcfg(self.target_cfg)
        self.grid_planner = gp
        # (len(self.vel_set), 3), each row i has the resulting r, theta, dtheta
        # encoding the change in position resulting from applying
        # self.vel_set[i, :] for self.rollout iterations to the wheelchair.
        # Stored in polar form so that they can easily be applied regardless
        # of where the wheelchair is facing (easy to add the angles).
        self.vel_rollout_deltas: np.ndarray = None

    def get_next_configs(self) -> List[Tuple[List[float], List[float]]]:
        if self.vel_rollout_deltas is None:
            self._init_vel_rollout_deltas()
        # Precompute the scores achieved assuming each primitive is feasible
        scores = np.empty(len(self.vel_rollout_deltas))
        curr_np = self.wu.cfg_to_rcfg(self.tracker.get_configs()[1])
        for i in range(len(self.vel_rollout_deltas)):
            vrd = self.vel_rollout_deltas[i, :]
            end_np = np.array([
                curr_np[0] + vrd[0] * np.cos(vrd[1] + curr_np[2]),
                curr_np[1] + vrd[0] * np.sin(vrd[1] + curr_np[2]),
                curr_np[2] + vrd[2]
            ])
            cfgs = self.tracker.get_configs()
            scores[i] = self.score_config_np(end_np)
            self.tracker.set_configs(cfgs)
        # sorts in increasing order
        score_sorted_inds = np.argsort(scores)
        # Eval in sorted order, pick first feasible
        for ind in score_sorted_inds:
            targ_vel = self.vel_set[ind, :]
            cfgs = self.tracker.get_configs()
            successful_rollout = True
            cfg_traj: List[Tuple[List[float], List[float]]] = []
            for _ in range(self.rollout):
                res = self.tracker.get_target_config(targ_vel)
                if res != "success":
                    successful_rollout = False
                    break
                cfgs = self.tracker.get_configs()
                part_score = self.score_config_np(self.wu.cfg_to_rcfg(cfgs[1]))
                cfg_traj.append(cfgs)
                # if part_score < scores[ind]:
                #     break
            if successful_rollout:
                print(self.vel_set[ind, :])
                return cfg_traj
            self.tracker.set_configs(cfgs)
        return None

    def score_config_np(self, wheelchair_np: np.ndarray) -> float:
        goal_dist_score = self.grid_planner.get_dist(wheelchair_np[:2])
        align_with_goal_score = 0
        if goal_dist_score < self.disp_tol:
            align_with_goal_score = abs(so2.diff(wheelchair_np[2],
                self.target_np[2]))
        self.tracker.wheelchair_model.setConfig(
            self.wu.rcfg_to_cfg(wheelchair_np))
        closest_dist = None
        max_dist = 2.0
        link_geo = self.tracker.wheelchair_model.link("base_link").geometry()
        for i in range(self.tracker.world_model.numTerrains()):
            terr: klampt.TerrainModel = self.tracker.world_model.terrain(i)
            if terr.getName() != "floor":
                if terr.geometry().withinDistance(link_geo, max_dist):
                    dist = terr.geometry().distance(link_geo).d
                    if closest_dist is None or dist < closest_dist:
                        closest_dist = dist
        if closest_dist is None:
            closest_dist = max_dist
        score = goal_dist_score + align_with_goal_score + 5*(max_dist - closest_dist)
        # print(align_with_goal_score, score)
        return score

    def _init_vel_rollout_deltas(self):
        self.vel_rollout_deltas = np.empty((len(self.vel_set), 3))
        for i in range(len(self.vel_set)):
            curr_pos = np.zeros(3)
            targ_vel = self.vel_set[i, :]
            # Forward Euler
            for _ in range(self.rollout):
                w_delta = np.array([
                    targ_vel[0] * np.cos(curr_pos[2]),
                    targ_vel[0] * np.sin(curr_pos[2]),
                    targ_vel[1]
                ]) * self.tracker.dt
                curr_pos += w_delta
            self.vel_rollout_deltas[i, :] = np.array([
                np.linalg.norm(curr_pos[:2]),
                np.arctan2(curr_pos[1], curr_pos[0]),
                curr_pos[2]
            ])
