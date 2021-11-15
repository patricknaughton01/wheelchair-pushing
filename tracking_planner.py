import json
from typing import Dict, List, Set, Tuple

from klampt.model import collide
from klampt.math import so2, vectorops as vo
from consts import SETTINGS_PATH
from planner import Planner
import numpy as np
import klampt
import heapq
from tracker import Tracker
from copy import deepcopy


class TrackingPlannerInstance(Planner):
    def __init__(self, world_model: klampt.WorldModel, dt: float):
        super().__init__(world_model, dt)
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.tracker_weights = {
            "arm_penalty": 0,
            "strafe_penalty": 1,
            "base_penalty": 0,
            "attractor_penalty": 10
        }
        self.rollout = 50
        self.robot_model: klampt.RobotModel = self.world_model.robot("trina")
        self.wheelchair_model: klampt.RobotModel = self.world_model.robot("wheelchair")
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]
        self.tracker = Tracker(self.world_model, self.dt, self.tracker_weights)
        self.executor = None

    def plan(self, target: np.ndarray, disp_tol: float, rot_tol: float):
        super().plan(target, disp_tol, rot_tol)
        cfg = self.tracker.wheelchair_model.getConfig()
        for i, d in enumerate(self.tracker.wheelchair_dofs):
            cfg[d] = target[i]
        self.executor = TrackerExecutor(self.tracker, cfg, self.disp_tol,
            self.rot_tol, rollout=self.rollout)
        # TODO: Add in call to TrackerExecutor for init config

    def next(self):
        self._check_target()
        # Check for termination:
        wheelchair_cfg = self.wheelchair_model.getConfig()
        wheelchair_xy = np.array([
            wheelchair_cfg[self.wheelchair_dofs[0]],
            wheelchair_cfg[self.wheelchair_dofs[1]]
        ])
        wheelchair_yaw = wheelchair_cfg[self.wheelchair_dofs[2]]
        if np.linalg.norm(wheelchair_xy - self.target[:2]) <= self.disp_tol:
            if so2.diff(wheelchair_yaw, self.target[2]) <= self.rot_tol:
                print("REACHED GOAL")
                raise StopIteration
        n_cfgs = self.executor.get_next_configs()
        if None in n_cfgs:
            print("NO VALID MOTIONS")
            raise StopIteration
        self.robot_model.setConfig(n_cfgs[0])
        self.wheelchair_model.setConfig(n_cfgs[1])


class TrackerExecutor:
    def __init__(self, tracker: Tracker, target_cfg: List[float],
        disp_tol: float, rot_tol: float,
        rollout: int=4, vel_set: np.ndarray=None, res: float=1.0,
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
            for fv in [0.1, 0.5, 1.0]:
                for rv in [0.0, 0.1, 0.5, 1.0]:
                    if not (fv < 1e-3 and rv < 1e-3):
                        vel_set_list.append([fv, rv])
                        vel_set_list.append([fv, -rv])
            vel_set = np.array(vel_set_list)
        self.vel_set = vel_set
        self.res = res
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]
        self.target_np = self._wheelchair_cfg_to_np(self.target_cfg)
        self.grid_planner = GridPlanner(self.tracker.world_model.copy(),
            self.target_cfg, self.res)
        # (len(self.vel_set), 3), each row i has the resulting r, theta, dtheta
        # encoding the change in position resulting from applying
        # self.vel_set[i, :] for self.rollout iterations to the wheelchair.
        # Stored in polar form so that they can easily be applied regardless
        # of where the wheelchair is facing (easy to add the angles).
        self.vel_rollout_deltas: np.ndarray = None

    def get_next_configs(self) -> Tuple[List[float], List[float]]:
        if self.vel_rollout_deltas is None:
            self._init_vel_rollout_deltas()
        # Precompute the scores achieved assuming each primitive is feasible
        scores = np.empty(len(self.vel_rollout_deltas))
        curr_np = self._wheelchair_cfg_to_np(self.tracker.get_configs()[1])
        for i in range(len(self.vel_rollout_deltas)):
            vrd = self.vel_rollout_deltas[i, :]
            end_np = np.array([
                curr_np[0] + vrd[0] * np.cos(vrd[1] + curr_np[2]),
                curr_np[1] + vrd[0] * np.sin(vrd[1] + curr_np[2]),
                curr_np[2] + vrd[2]
            ])
            scores[i] = self.score_config_np(end_np)
        # sorts in increasing order
        score_sorted_inds = np.argsort(scores)
        # Eval in sorted order, pick first feasible
        for ind in score_sorted_inds:
            targ_vel = self.vel_set[ind, :]
            cfgs = self.tracker.get_configs()
            successful_rollout = True
            for _ in range(self.rollout):
                res = self.tracker.get_target_config(targ_vel)
                if res != "success":
                    successful_rollout = False
                    break
            if successful_rollout:
                return self.tracker.get_configs()
            self.tracker.set_configs(cfgs)
        return None, None

    def score_config_np(self, wheelchair_np: np.ndarray) -> float:
        goal_dist_score = self.grid_planner.get_dist(wheelchair_np[:2])
        align_with_goal_score = 0
        if goal_dist_score < self.disp_tol:
            align_with_goal_score = abs(so2.diff(wheelchair_np[2],
                self.target_np[2]))
        return goal_dist_score + align_with_goal_score

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

    def _wheelchair_cfg_to_np(self, cfg: List[float]) -> np.ndarray:
        arr = []
        for d in self.wheelchair_dofs:
            arr.append(cfg[d])
        return np.array(arr)


class GridPlanner:
    def __init__(self, world_model: klampt.WorldModel,
        target_cfg: List[float], res: float
    ):
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        # self.world_model = world_model
        self.world_model = klampt.WorldModel()
        self.world_model.loadFile("Model/worlds/TRINA_world_cholera.xml")
        self.robot_model: klampt.RobotModel = self.world_model.robot("wheelchair")
        self.base_name = "base_link"
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]
        self.target_cfg = target_cfg
        self.res = res
        self.collider = collide.WorldCollider(self.world_model)
        self._ignore_collision_pairs()
        self._set_collision_margins()
        self.robot_model.setConfig(self.target_cfg)
        self.target_pos = np.array([
            self.target_cfg[self.wheelchair_dofs[0]],
            self.target_cfg[self.wheelchair_dofs[1]]
        ])
        self.yaw = self.target_cfg[self.wheelchair_dofs[2]]
        self.cache: Dict[Tuple[int, int], float] = {}
        start_ind = self._pos_to_ind(self.target_pos)
        self.open_set: List[Tuple[float, float, Tuple[int, int]]] = [(0, 0, start_ind)]
        self.open_d_map: Dict[Tuple[int, int], float] = {}
        self.deltas = [(0,1), (0,-1), (-1,0), (1,0), (1,1), (1,-1), (-1,1), (-1,-1)]

    def get_dist(self, pos: np.ndarray) -> float:
        ind = list(self._pos_to_ind(pos))
        corner_inds: List[Tuple[int, int]] = []
        for delta in [[0,0], [0,1], [1,0], [1,1]]:
            corner_inds.append(tuple(vo.add(ind, delta)))
            if self._collides(corner_inds[-1]):
                self.cache[corner_inds[-1]] = float('inf')
        # A* to find grid distance to the relevant four corners
        while (not self._found_corners(corner_inds)) and (len(self.open_set) > 0):
            _, min_dist, min_ind = heapq.heappop(self.open_set)
            # Only continue if we haven't already expanded this index
            if min_ind not in self.cache:
                # print("Expanding", min_ind)
                self.cache[min_ind] = min_dist
                for delta in self.deltas:
                    n_ind = (min_ind[0] + delta[0], min_ind[1] + delta[1])
                    if n_ind in self.cache:
                        continue
                    cand_cost = self._cost(min_ind, n_ind) + min_dist
                    best_known_dist = self.open_d_map.get(n_ind, float('inf'))
                    if cand_cost < best_known_dist:
                        heapq.heappush(self.open_set,
                            (self._heuristic(n_ind, pos) + cand_cost,
                                cand_cost, n_ind))
                        self.open_d_map[n_ind] = cand_cost
        return self._bilinear_interpolate(corner_inds, pos)

    def _found_corners(self, corner_inds: List[Tuple[int, int]]) -> bool:
        for c in corner_inds:
            if c not in self.cache:
                return False
        return True

    def _bilinear_interpolate(self, corner_inds: List[Tuple[int, int]], pos: np.ndarray) -> float:
        # Assumes corners come in order:
        # 0 2
        # 1 3
        if not self._found_corners(corner_inds):
            raise RuntimeError("Not all corners have known distances")
        dists = []
        poses = []
        for c in corner_inds:
            dists.append(self.cache[c])
            poses.append(self._ind_to_pos(c))
        frac_x = (pos[0] - poses[0][0]) / (poses[2][0] - poses[0][0])
        frac_y = (pos[1] - poses[0][1]) / (poses[1][1] - poses[0][1])
        t1 = dists[0] * (1 - frac_x) + dists[2] * frac_x
        t2 = dists[1] * (1 - frac_x) + dists[3] * frac_x
        ret = t1 * (1 - frac_y) + t2 * frac_y
        return ret

    def _cost(self, inda: Tuple[int, int], indb: Tuple[int, int]) -> float:
        for ind in (inda, indb):
            if self._collides(ind):
                return float('inf')
        na = np.array(inda) * self.res
        nb = np.array(indb) * self.res
        return np.linalg.norm(nb - na)

    def _collides(self, ind: Tuple[int, int]) -> bool:
        self.robot_model.setConfig(self._ind_to_cfg(ind))
        collides = False
        for c in self.collider.collisions():
            print(c[0].getName(), c[1].getName())
            collides = True
            break
        return collides

    def _heuristic(self, ind: Tuple[int, int], pos: np.ndarray) -> float:
        return 0

    def _pos_to_ind(self, pos: np.ndarray) -> Tuple[int, int]:
        return (int(pos[0] // self.res), int(pos[1] // self.res))

    def _ind_to_pos(self, ind: Tuple[int, int]) -> np.ndarray:
        return np.array([ind[0] * self.res, ind[1] * self.res])

    def _ind_to_cfg(self, ind: Tuple[int, int]) -> List[float]:
        pos = self._ind_to_pos(ind)
        cfg = self.robot_model.getConfig()
        cfg[self.wheelchair_dofs[0]] = pos[0]
        cfg[self.wheelchair_dofs[1]] = pos[1]
        cfg[self.wheelchair_dofs[2]] = self.yaw
        return cfg

    def _ignore_collision_pairs(self):
        for i in range(self.robot_model.numLinks()):
            for j in range(self.robot_model.numLinks()):
                if i != j:
                    link_a = self.robot_model.link(i)
                    link_b = self.robot_model.link(j)
                    if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                        self.collider.ignoreCollision((link_a, link_b))
        # For this planner, ignore collisions between the wheelchair and
        # trina, just want to check against obstacles
        trina_model: klampt.RobotModel = self.world_model.robot("trina")
        for i in range(self.robot_model.numLinks()):
            for j in range(trina_model.numLinks()):
                link_a = self.robot_model.link(i)
                link_b = trina_model.link(j)
                if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                    self.collider.ignoreCollision((link_a, link_b))
        # Ignore TRINA self collisions
        for i in range(trina_model.numLinks()):
            for j in range(trina_model.numLinks()):
                if i != j:
                    link_a = trina_model.link(i)
                    link_b = trina_model.link(j)
                    if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                        self.collider.ignoreCollision((link_a, link_b))
        # Ignore any collision with the floor
        self.collider.ignoreCollision(self.world_model.terrain("floor"))

    def _set_collision_margins(self):
        for i in range(self.robot_model.numLinks()):
            link = self.robot_model.link(i)
            if not link.geometry().empty():
                link.geometry().setCollisionMargin(0.5)


if __name__ == "__main__":
    import time
    from klampt import vis
    world = klampt.WorldModel()
    world.loadFile("Model/worlds/TRINA_world_cholera.xml")
    robot_model = world.robot("trina")
    wheelchair_model = world.robot("wheelchair")
    vis.add("world", world)
    vis.show()
    dt = 1 / 50
    planner = TrackingPlannerInstance(world.copy(), dt)
    planner.plan(np.array([10.0, 0.0, 0.0]), 0.5, 0.5)
    iter = 0
    while vis.shown():
        iter += 1
        try:
            planner.next()
            robot_model.setConfig(planner.robot_model.getConfig())
            wheelchair_model.setConfig(planner.wheelchair_model.getConfig())
        except StopIteration:
            print("Stopped at iteration ", iter)
            break
