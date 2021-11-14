import json
from typing import Dict, List, Set, Tuple

from klampt.model import collide
from klampt.math import vectorops as vo
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
        self.tracker_weights = {
            "arm_penalty": 0,
            "strafe_penalty": 1,
            "base_penalty": 0,
            "attractor_penalty": 10
        }
        self.robot_model = self.world_model.robot("trina")
        self.wheelchair_model = self.world_model.robot("wheelchair")
        self.tracker = Tracker(self.world_model, self.dt, self.tracker_weights)
        self.executor = None

    def plan(self, target: np.ndarray, disp_tol: float, rot_tol: float):
        super().plan(target, disp_tol, rot_tol)
        cfg = self.tracker.wheelchair_model.getConfig()
        for i, d in enumerate(self.tracker.wheelchair_dofs):
            cfg[d] = target[i]
        self.executor = TrackerExecutor(self.tracker, cfg, self.disp_tol,
            self.rot_tol)

    def next(self):
        self._check_target()
        n_cfgs = self.executor.get_next_configs()
        if None in n_cfgs:
            raise StopIteration
        self.robot_model.setConfig(n_cfgs[0])
        self.wheelchair_model.setConfig(n_cfgs[1])


class TrackerExecutor:
    def __init__(self, tracker: Tracker, target_cfg: List[float],
        disp_tol: float, rot_tol: float, vel_set: np.ndarray=None,
        res: float=0.1
    ):
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.tracker = tracker
        self.target_cfg = target_cfg
        self.disp_tol = disp_tol
        self.rot_tol = rot_tol
        if vel_set is None:
            vel_set = np.array([
                [0.0, 1.0],
                [0.5, 1.0],
                [0.7, 0.5],
                [1.0, 0.0],
                [0.7, -0.5],
                [0.5, -1.0],
                [0.0, -1.0]
            ])
        self.vel_set = vel_set
        self.res = res
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]
        self.grid_planner = GridPlanner(self.tracker.world_model.copy(),
            self.target_cfg, self.res)

    def get_next_configs(self) -> Tuple[List[float], List[float]]:
        best_score = None   # Low score is better
        best_cfgs = None
        for i in range(len(self.vel_set)):
            targ_vel = self.vel_set[i, :]
            print("Testing velocity", targ_vel)
            cfgs = self.tracker.get_configs()
            res = self.tracker.get_target_config(targ_vel)
            if res == "success":
                score = self.score_config()
                print("Score: ", score)
                if best_score is None or score < best_score:
                    best_score = score
                    best_cfgs = self.tracker.get_configs()
            else:
                print("Candidate twist ", self.vel_set[i, :], " failed with error ", res)
            self.tracker.set_configs(cfgs)
        if best_score is not None:
            return best_cfgs
        return None, None

    def score_config(self) -> float:
        print("SCORING CONFIG")
        cfg = self.tracker.wheelchair_model.getConfig()
        return self.grid_planner.get_dist(
            np.array([cfg[self.wheelchair_dofs[0]],
                cfg[self.wheelchair_dofs[1]]]))


class GridPlanner:
    def __init__(self, world_model: klampt.WorldModel,
        target_cfg: List[float], res: float
    ):
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.world_model = world_model
        self.robot_model = self.world_model.robot("wheelchair")
        self.base_name = "base_link"
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]
        self.target_cfg = target_cfg
        self.res = res
        self.collider = collide.WorldCollider(self.world_model)
        self._ignore_collision_pairs()
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
        self.deltas = [(0,1), (0,-1), (-1,0), (-1,0)]

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
        # print(corner_inds)
        dists = []
        poses = []
        for c in corner_inds:
            dists.append(self.cache[c])
            poses.append(self._ind_to_pos(c))
        # print(dists)
        # print(poses)
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
        for _ in self.collider.collisions():
            collides = True
            break
        return collides

    def _heuristic(self, ind: Tuple[int, int], pos: np.ndarray) -> float:
        return np.linalg.norm(np.array(ind) * self.res - pos)

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
        trina_model = self.world_model.robot("trina")
        for i in range(self.robot_model.numLinks()):
            for j in range(trina_model.numLinks()):
                link_a = self.robot_model.link(i)
                link_b = trina_model.link(j)
                if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                    self.collider.ignoreCollision((link_a, link_b))
        # Ignore TRINA collisions
        for i in range(trina_model.numLinks()):
            for j in range(trina_model.numLinks()):
                if i != j:
                    link_a = trina_model.link(i)
                    link_b = trina_model.link(j)
                    if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                        self.collider.ignoreCollision((link_a, link_b))


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
    planner.plan(np.array([10.0, 0.0, 0.0]), 0.1, 0.1)
    iter = 0
    while True:
        iter += 1
        try:
            planner.next()
            robot_model.setConfig(planner.robot_model.getConfig())
            wheelchair_model.setConfig(planner.wheelchair_model.getConfig())
            print()
            time.sleep(dt)
        except StopIteration:
            print("Stopped at iteration ", iter)
            break
