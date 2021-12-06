import json
import heapq
import math
from typing import Dict, List, Tuple
import klampt
from klampt.model import collide
from klampt.math import so2, vectorops as vo
import numpy as np
from utils import WheelchairUtility

from consts import SETTINGS_PATH


class GridPlanner:
    def __init__(self, world_fn: str,
        target_cfg: List[float], res: float, rot_res: float=np.pi/4,
        rot_d_scale: float=0.1
    ):
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.world_model = klampt.WorldModel()
        self.world_model.loadFile(world_fn)
        self.robot_model: klampt.RobotModel = self.world_model.robot("wheelchair")
        self.wu = WheelchairUtility(self.robot_model)
        self.base_name = "base_link"
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]
        self.target_cfg = target_cfg
        self.res = res
        self.rot_res = rot_res
        self.rot_d_scale = rot_d_scale
        self.num_angle_inds = math.floor(2 * np.pi / self.rot_res)
        self.collider = collide.WorldCollider(self.world_model)
        self._ignore_collision_pairs()
        self._set_collision_margins()
        self.robot_model.setConfig(self.target_cfg)
        self.target_np = self.wu.cfg_to_rcfg(self.target_cfg)
        self.yaw = self.target_cfg[self.wheelchair_dofs[2]]
        self.cache: Dict[Tuple[int, int, int], Tuple[float, Tuple[int, int, int]]] = {}
        start_ind = self._pos_to_ind(self.target_np)
        self.open_set: List[Tuple[float, float, Tuple[int, int, int], Tuple[int, int, int]]] = [(0, 0, start_ind, start_ind)]
        self.open_d_map: Dict[Tuple[int, int, int], float] = {}

    def get_dist(self, pos: np.ndarray) -> float:
        tup_ind = self._pos_to_ind(pos)
        ind = list(tup_ind)
        # print("GRID PLANNER QUERIED IND: ", ind, pos)
        corner_inds: List[Tuple[int, int, int]] = []
        for delta in [[0,0,0]]:#, [0,1,0], [1,0,0], [1,1,0], [0,0,1], [0,1,1], [1,0,1], [1,1,1]]:
            neighbor = vo.add(ind, delta)
            neighbor[2] = neighbor[2] % self.num_angle_inds
            corner_inds.append(tuple(neighbor))
        # Skip updating the open set if we already have all necessary info
        # for this query
        if not self._found_corners(corner_inds):
            # Update heuristic values in open set based on new pos query
            for i, val in enumerate(self.open_set):
                self.open_set[i] = (
                    self._heuristic(val[2], pos) + val[1], val[1], val[2], val[3]
                )
            heapq.heapify(self.open_set)
        # A* to find grid distance to the relevant four corners
        while (not self._found_corners(corner_inds)) and (len(self.open_set) > 0):
            _, min_dist, min_ind, bp = heapq.heappop(self.open_set)
            # Only continue if we haven't already expanded this index
            if min_ind not in self.cache:
                self.cache[min_ind] = (min_dist, bp)
                for n_ind, c in self._neighbors(min_ind):
                    if n_ind in self.cache:
                        continue
                    cand_cost = c + min_dist
                    best_known_dist = self.open_d_map.get(n_ind, float('inf'))
                    if cand_cost < best_known_dist:
                        heapq.heappush(self.open_set,
                            (self._heuristic(n_ind, pos) + cand_cost,
                                cand_cost, n_ind, min_ind))
                        self.open_d_map[n_ind] = cand_cost
        # c1 = self._bilinear_interpolate(corner_inds[:4], pos)
        # c2 = self._bilinear_interpolate(corner_inds[4:], pos)
        # yaw = pos[2]
        # yaw_ind = ind[2]
        # n_yaw_ind = (ind[2] + 1) % self.num_angle_inds
        # yaw_ind_yaw = yaw_ind * self.rot_res
        # n_yaw_ind_yaw = n_yaw_ind * self.rot_res
        # bd1 = so2.diff(yaw, yaw_ind_yaw)
        # d = so2.diff(n_yaw_ind_yaw, yaw_ind_yaw)
        # ret = c1 * (1 - bd1 / d) + c2 * bd1 / d
        # print(c1, c2, yaw, yaw_ind_yaw, n_yaw_ind_yaw)
        # ret = self.cache[tup_ind][0]
        return self.cache[tup_ind]


    def _found_corners(self, corner_inds: List[Tuple[int, int, int]]) -> bool:
        for c in corner_inds:
            if c not in self.cache:
                return False
        return True

    def _bilinear_interpolate(self, corner_inds: List[Tuple[int, int, int]], pos: np.ndarray) -> float:
        # Assumes corners come in order:
        # 0 2
        # 1 3
        if not self._found_corners(corner_inds):
            raise RuntimeError("Not all corners have known distances")
        dists = []
        poses = []
        for c in corner_inds:
            dists.append(self.cache[c][0])
            poses.append(self._ind_to_pos(c))
        frac_x = (pos[0] - poses[0][0]) / (poses[2][0] - poses[0][0])
        frac_y = (pos[1] - poses[0][1]) / (poses[1][1] - poses[0][1])
        t1 = dists[0] * (1 - frac_x) + dists[2] * frac_x
        t2 = dists[1] * (1 - frac_x) + dists[3] * frac_x
        ret = t1 * (1 - frac_y) + t2 * frac_y
        return ret

    def _neighbors(self, s: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], float]]:
        x_ind, y_ind, yaw_ind = s
        pos = self._ind_to_pos(s)
        yaw = pos[2]
        # Turn in place
        neighbors = [
            ((x_ind, y_ind, (yaw_ind-1) % self.num_angle_inds), self.rot_d_scale * self.rot_res),
            ((x_ind, y_ind, (yaw_ind+1) % self.num_angle_inds), self.rot_d_scale * self.rot_res)
        ]
        # Go backwards from node (because expanding from goal)
        new_x_ind = round(x_ind - np.cos(yaw))
        new_y_ind = round(y_ind - np.sin(yaw))
        new_ind = (new_x_ind, new_y_ind, yaw_ind)
        new_pos = self._ind_to_pos(new_ind)
        neighbors.append(
            (new_ind, np.linalg.norm(new_pos[:2] - pos[:2]))
        )
        max_dist = 0.75
        link_geo = self.robot_model.link("base_link").geometry()
        for i in range(len(neighbors)):
            ind, c = neighbors[i]
            pos = self._ind_to_pos(ind)
            cfg = self.wu.rcfg_to_cfg(pos)
            self.robot_model.setConfig(cfg)
            closest_dist = None
            for j in range(self.world_model.numTerrains()):
                terr: klampt.TerrainModel = self.world_model.terrain(j)
                if terr.getName() == "floor":
                    continue
                if terr.geometry().withinDistance(link_geo, max_dist):
                    dist = terr.geometry().distance(link_geo).d
                    if closest_dist is None or dist < closest_dist:
                        closest_dist = dist
            if closest_dist is not None:
                c += np.exp(-closest_dist)
                if closest_dist < 0:
                    c = float('inf')
                    # c += 10 * np.exp(-closest_dist)
            neighbors[i] = (ind, c)
        return neighbors

    def _collides(self, ind: Tuple[int, int, int]) -> bool:
        self.robot_model.setConfig(self._ind_to_cfg(ind))
        collides = False
        for _ in self.collider.collisions():
            collides = True
            break
        return collides

    def _heuristic(self, ind: Tuple[int, int, int], pos: np.ndarray) -> float:
        # return 0
        return np.linalg.norm(self._ind_to_pos(ind)[:2] - pos[:2])

    def _pos_to_ind(self, pos: np.ndarray) -> Tuple[int, int, int]:
        yaw = pos[2]
        while yaw >= 2 * np.pi:
            yaw -= 2 * np.pi
        while yaw < 0:
            yaw += 2 * np.pi
        yaw_ind = math.floor(yaw / self.rot_res) % self.num_angle_inds
        return (
            math.floor(pos[0] / self.res),
            math.floor(pos[1] / self.res),
            yaw_ind
        )

    def _ind_to_pos(self, ind: Tuple[int, int, int]) -> np.ndarray:
        return np.array([
            ind[0] * self.res,
            ind[1] * self.res,
            (ind[2] % self.num_angle_inds) * self.rot_res
        ])

    def _ind_to_cfg(self, ind: Tuple[int, int, int]) -> List[float]:
        pos = np.array(self._ind_to_pos(ind))
        o_cfg = self.robot_model.getConfig()
        cfg = self.wu.rcfg_to_cfg(pos)
        self.robot_model.setConfig(o_cfg)
        return cfg

    def _pos_to_nearest_pos(self, pos: np.ndarray) -> np.ndarray:
        ind = self._pos_to_ind(pos)
        closest_dist = None
        closest_xy = None
        closest_yaw = None
        for delta in [[0,0,0], [0,1,0], [1,0,0], [1,1,0]]:
            neighbor = vo.add(ind, delta)
            neighbor[2] = neighbor[2] % self.num_angle_inds
            n_pos = self._ind_to_pos(neighbor)
            dist = np.linalg.norm(n_pos[:2] - pos[:2])
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_xy = n_pos[:2]
        yaw = pos[2]
        yaw_ind = ind[2]
        n_yaw_ind = (ind[2] + 1) % self.num_angle_inds
        yaw_ind_yaw = yaw_ind * self.rot_res
        n_yaw_ind_yaw = n_yaw_ind * self.rot_res
        if abs(so2.diff(yaw, yaw_ind_yaw)) < abs(so2.diff(n_yaw_ind_yaw, yaw)):
            closest_yaw = yaw_ind_yaw
        else:
            closest_yaw = n_yaw_ind_yaw
        return np.array([*closest_xy, closest_yaw])

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
                link.geometry().setCollisionMargin(0.25)
