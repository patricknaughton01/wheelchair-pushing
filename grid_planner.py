import json
import heapq
from typing import Dict, List, Tuple
import klampt
from klampt.model import collide
from klampt.math import vectorops as vo
import numpy as np

from consts import SETTINGS_PATH


class GridPlanner:
    def __init__(self, world_fn: str,
        target_cfg: List[float], res: float
    ):
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.world_model = klampt.WorldModel()
        self.world_model.loadFile(world_fn)
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
        for _ in self.collider.collisions():
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
                link.geometry().setCollisionMargin(0.75)
