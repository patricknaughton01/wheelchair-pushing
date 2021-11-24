from typing_extensions import Concatenate
import numpy as np
from typing import List, Dict, Tuple
from utils.state_lattice import *
from utils.helper import *
from utils.tw import *
import heapq
import json
from consts import SETTINGS_PATH
from planner import Planner
import klampt
from klampt.math import se3, so3
from klampt.model import ik, collide
from grid_planner import GridPlanner
from tracker import mirror_arm_config

class StateLatticePlanner(Planner):
    """Planning with state lattice
    """
    def __init__(self, sl:StateLattice, world_fn: str, dt: float):
        super().__init__(world_fn, dt)
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)

        self.dt = dt
        self.left_name = "left_tool_link"
        self.right_name = "right_tool_link"
        self.base_name = "base_link"
        self.left_handle_name = "lefthandle_link"
        self.right_handle_name = "righthandle_link"
        self.w_base_name = "base_link"
        self.robot_model: klampt.RobotModel = self.world_model.robot("trina")
        self.wheelchair_model: klampt.RobotModel = self.world_model.robot("wheelchair")
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]

        self.w_t_bl = se3.mul(
            se3.inv(
                self.wheelchair_model.link(self.w_base_name).getTransform()),
            self.wheelchair_model.link(self.left_handle_name).getTransform())
        self.w_t_br = se3.mul(
            se3.inv(
                self.wheelchair_model.link(self.w_base_name).getTransform()),
            self.wheelchair_model.link(self.right_handle_name).getTransform())
        self.t_hee = self._get_t_hee()
        self.left_dofs: List[int] = self.settings["left_arm_dofs"]
        self.right_dofs: List[int] = self.settings["right_arm_dofs"]
        self.left_gripper_dofs: List[int] = self.settings["left_gripper_dofs"]
        self.right_gripper_dofs: List[int] = self.settings["right_gripper_dofs"]
        self.base_dofs: List[int] = self.settings["base_dofs"]
        self.num_arm_dofs = len(self.left_dofs) + len(self.right_dofs)
        self.num_total_dofs = self.num_arm_dofs + len(self.base_dofs)
        self._init_config()
        self.t_lr = se3.mul(se3.inv(
            self.robot_model.link(self.left_name).getTransform()),
            self.robot_model.link(self.right_name).getTransform())

        self.collider = collide.WorldCollider(self.world_model)
        self._ignore_collision_pairs()
        self._set_collision_margins()
        self.cfg_ind = 0

        self.sl = sl
        self.close_set: Dict[Tuple[int, int, int, int], float] = {}
        self.start_idx = (0, 0, 3, 3)
        self.open_set: List[Tuple[float, float, Tuple[int, int, int, int]]] = [(0,0,self.start_idx)] # start from the origin point
        self.open_d_map: Dict[Tuple[int, int, int, int], float] = {}
        self.traj = {}
        self.d = self.sl.sys.sets['d']

    def plan(self, tgt: np.ndarray, disp_tol: float, rot_tol: float):
        """get the planned trajectory with SL

        Args:
            tgt (List): target

        Returns:
            list: [trajectory data]
        """
        print("start plan")
        self.tgt = tgt
        self.tgt_idx = self._pos_to_ind(tgt)
        super().plan(tgt, disp_tol, rot_tol)
        cfg = self._wheelchair_np_to_cfg(tgt)
        gp = GridPlanner(self.world_fn, cfg, self.sl.r)
        # warm start the grid planner
        gp.get_dist(tgt)
        if self._collides(self.tgt_idx):
            self.close_set[self.tgt_idx] = float('inf')
        while (self.tgt_idx not in self.close_set) and len(self.open_set) > 0:
            _, min_dist, min_ind = heapq.heappop(self.open_set)
            if min_ind not in self.close_set:
                self.close_set[min_ind] = min_dist
                suc_nodes, suc_data, orien_idx = self.get_successor(min_ind)
                for i, n_ind in enumerate(suc_nodes):
                    if n_ind in self.close_set:
                        continue
                    cand_cost = suc_data['costs'][i] + min_dist + self._cost_obs(n_ind)
                    best_known_dist = self.open_d_map.get(n_ind, float('inf'))
                    if cand_cost < best_known_dist:
                        n_pos = self._ind_to_pos(n_ind)
                        # euclidean heuristic
                        # heapq.heappush(self.open_set,(self.euclidean_heuristic(n_ind) + cand_cost, cand_cost, n_ind))
                        # Dijkstra heuristic
                        heapq.heappush(self.open_set,(gp.get_dist(n_pos) + cand_cost, cand_cost, n_ind))
                        self.open_d_map[n_ind] = cand_cost
                        self.traj[n_ind] = {'prev': min_ind, 'state_w': suc_data['states_w'][i], 'state_r':suc_data['states_r'][i], 'action_taken': [orien_idx, i]}
        return self.retrive_traj()

    def next(self):
        # self._check_target()
        # Check for termination:
        wheelchair_cfg = self.wheelchair_model.getConfig()
        wheelchair_xy = np.array([
            wheelchair_cfg[self.wheelchair_dofs[0]],
            wheelchair_cfg[self.wheelchair_dofs[1]]
        ])
        wheelchair_yaw = wheelchair_cfg[self.wheelchair_dofs[2]]
        if np.linalg.norm(wheelchair_xy - self.target[:2]) <= self.disp_tol:
            if abs(so2.diff(wheelchair_yaw, self.target[2])) <= self.rot_tol:
                raise StopIteration
        # gen config
        res = self.get_target_config(self.traj_w[self.cfg_ind, :], self.traj_r[self.cfg_ind, :])
        if res != "success":
            raise StopIteration
        self.cfg_ind += 1
        
    def get_target_config(self, w_np, r_np) -> List[float]:
        w_t_wb = self.wheelchair_model.link(self.w_base_name).getTransform()
        w_q = _wheelchair_np_to_cfg(w_np)
        self.wheelchair_model.setConfig(w_q)
        collides = False
        for _ in self.collider.collisions():
            collides = True
            break
        if collides:
            return "collision"
        # In world frame, translation vectors from wheelchair base to handles
        r_q = _robot_np_to_cfg(r_np)
        self.robot_model.setConfig(r_q)

        for i in range(2):
            if i == 0:
                handle_name = self.left_handle_name
                link = self.robot_model.link(self.left_name)
                dofs = self.left_dofs
            elif i == 1:
                handle_name = self.right_handle_name
                link = self.robot_model.link(self.right_name)
                dofs = self.right_dofs
            else:
                break
            t_wee = se3.mul(
                self.wheelchair_model.link(handle_name).getTransform(),
                self.t_hee)
            goal = ik.objective(link, R=t_wee[0], t=t_wee[1])
            if not ik.solve_nearby(goal, 1, activeDofs=dofs):
                return "ik"
        collides = False
        for _ in self.collider.collisions():
            collides = True
            break
        if self.robot_model.selfCollides() or collides:
            print("collision detected!")
            return "collision"
        return "success"

    def get_configs(self) -> Tuple[List[float], List[float]]:
        return self.robot_model.getConfig(), self.wheelchair_model.getConfig()

    def set_configs(self, cfgs: Tuple[List[float], List[float]]):
        self.robot_model.setConfig(cfgs[0])
        self.wheelchair_model.setConfig(cfgs[1])

    def get_successor(self, node: Tuple[int, int, int, int]):
        """get successor for a given node

        Args:
            node (Tuple[int, int, int, int]): query node

        Returns:
            list: successor node, trajectories and index of the SL orientation (0 to 7)
        """
        suc_nodes = []
        cur_node = (0,0,node[2],node[3])
        orien_idx = self.sl.keys.index(cur_node)//3
        suc_data = self.sl.data[cur_node]
        for inc in suc_data['targets']:
            suc_node = (node[0] + inc[0], node[1] + inc[1], inc[2], inc[3])
            suc_nodes.append(suc_node)
        return suc_nodes, suc_data, orien_idx

    def retrive_traj(self):
        end = self.tgt_idx
        nodes, states_w, states_r, action_taken = [self.tgt], [], [], []
        while self.traj[end]['prev'] != self.start_idx:
            prev = self.traj[end]['prev']
            prev_w_pose = self._ind_to_pos(prev)
            prev_r_pose = self.get_init_robot_state(prev_w_pose)
            nodes.insert(0,list(prev_w_pose))
            states_w.insert(0,self.state_l2g(prev_w_pose, self.traj[end]['state_w']))
            states_r.insert(0,self.state_l2g(prev_r_pose, self.traj[end]['state_r']))
            action_taken.insert(0,self.traj[end]['action_taken'])
            end = prev
        print(f"nodes length: {len(nodes)}")
        self.traj_w = np.asarray(states_w).reshape(-1, 3)
        self.traj_r = np.asarray(states_r).reshape(-1, 3)
        # self.gen_cfg_buffer(traj_w, traj_r)
        return np.asarray(nodes), self.traj_w, self.traj_r, np.asarray(action_taken)

    # def gen_cfg_buffer(self, traj_w, traj_r):
    def _cost_obs(self, ind: Tuple[int, int]) -> float:
        if self._collides(ind):
            return float('inf')
        return 0

    def state_l2g(self, origin, state):
        """shift state with respect to the origin

        Args:
            origin (list): new origin
            state (ndarray): state to transform

        Returns:
            ndarray: transformed states
        """
        xy = state[:,0:2] + origin[0:2]
        psi = []
        for i in range(state.shape[0]):
            psi.append(diff_angle(state[i, 2], -origin[2]))
        psi = np.asarray(psi).reshape(-1,1)
        return np.concatenate((xy, psi), axis = 1).tolist()

    def get_init_robot_state(self, cur_w_pose):
        """extract robot pos form wheelchar state

        Args:
            cur_w_pose (list/ndarray): wheelchair state

        Returns:
            list: robot state
        """
        cur_r_pose = [0,0,cur_w_pose[3]]
        cur_r_pose[0] = cur_w_pose[0] - self.d*np.cos(cur_w_pose[3])
        cur_r_pose[1] = cur_w_pose[1] - self.d*np.sin(cur_w_pose[3])
        return cur_r_pose
        
    def euclidean_heuristic(self, node):
        cur_pos = self._ind_to_pos(node)
        dist = cur_pos[0:2] - self.tgt[0:2]
        return np.linalg.norm(dist)

    def _heuristic(self, ind: Tuple[int, int], pos: np.ndarray) -> float:
        return 0 # TODO: query from P's Dijkstra

    def _pos_to_ind(self, pos: List) -> Tuple[int, int, int, int]:
        return (int(pos[0] // self.sl.r), int(pos[1] // self.sl.r), \
                int(pos[2] // self.sl.delta_psi) + 3, int(pos[3] // self.sl.delta_psi) + 3)

    def _ind_to_pos(self, ind: Tuple[int, int, int, int]) -> np.ndarray:
        return self.sl.idx2pos(ind)
    
    def _ind_to_cfg(self, ind: Tuple[int, int, int, int]) -> List[float]:
        pos = self._ind_to_pos(ind)
        cfg = self.wheelchair_model.getConfig()
        cfg[self.wheelchair_dofs[0]] = pos[0]
        cfg[self.wheelchair_dofs[1]] = pos[1]
        cfg[self.wheelchair_dofs[2]] = pos[2]
        return cfg

    def _collides(self, ind: Tuple[int, int, int, int]) -> bool:
        self.wheelchair_model.setConfig(self._ind_to_cfg(ind))
        collides = False
        for _ in self.collider.collisions():
            collides = True
            break
        return collides
    
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

    def _wheelchair_cfg_to_np(self, cfg: List[float]) -> np.ndarray:
        arr = []
        for d in self.wheelchair_dofs:
            arr.append(cfg[d])
        return np.array(arr)

    def _wheelchair_np_to_cfg(self, w_np: List[float]) -> List[float]:
        cfg = self.wheelchair_model.getConfig()
        for i, d in enumerate(self.wheelchair_dofs):
            cfg[d] = w_np[i]
        return cfg

    def _robot_np_to_cfg(self, r_np: List[float]) -> List[float]:
        cfg = self.robot_model.getConfig()
        for i, d in enumerate(self.base_dofs):
            cfg[d] = r_np[i]
        return cfg

    def _get_t_hee(self) -> tuple:
        # Hand tilted 45 degrees down toward the wheelchair handle
        aa = ((0, 1, 0), np.pi / 4)
        r_hee = so3.from_axis_angle(aa)
        return (r_hee, (-0.05, 0, 0.15))

    def _init_config(self):
        """Set the robot's config to grasp the handles of the wheelchair.
        """
        t_wee = se3.mul(
            self.wheelchair_model.link(self.left_handle_name).getTransform(),
            self.t_hee)
        self.set_arm_cfg('l', self.settings["configs"]["home"])
        goal = ik.objective(self.robot_model.link(self.left_name),
            R=t_wee[0], t=t_wee[1])
        if not ik.solve(goal, activeDofs=self.left_dofs):
            raise RuntimeError("Couldn't find initial grasp pose, quitting")
        l_cfg = self.get_arm_cfg('l', self.robot_model.getConfig()).tolist()
        r_cfg = mirror_arm_config(l_cfg)
        self.set_arm_cfg('l', l_cfg)
        self.set_arm_cfg('r', r_cfg)

    def get_arm_cfg(self, side: str, cfg: List[float]) -> np.ndarray:
        if side.lower().startswith("l"):
            return np.array(cfg)[self.left_dofs]
        elif side.lower().startswith("r"):
            return np.array(cfg)[self.right_dofs]
        raise ValueError(f"Side {side} invalid")

    def set_arm_cfg(self, side: str, cfg: List[float]):
        if side.lower().startswith("l"):
            dofs = self.left_dofs
        elif side.lower().startswith("r"):
            dofs = self.right_dofs
        else:
            raise ValueError(f"Side {side} invalid")
        if len(cfg) != len(dofs):
            raise ValueError(f"Config size incorrect, {len(cfg)}!={len(dofs)}")
        q = self.robot_model.getConfig()
        for i, d in enumerate(dofs):
            q[d] = cfg[i]
        self.robot_model.setConfig(q)

if __name__ == "__main__":
    import time
    from klampt import vis
    world_fn = "Model/worlds/world_short_turn.xml"
    world = klampt.WorldModel()
    world.loadFile(world_fn)
    robot_model = world.robot("trina")
    wheelchair_model = world.robot("wheelchair")

    dt = 1 / 50
    cfg = Config("/home/yu/Documents/courses/KKH598/proj/state_lattice_planner/data/params/tw.yaml")
    sys = TWSys(cfg.value, seed = 0)
    sl = StateLattice(sys)
    logs = sl.load("/home/yu/Documents/courses/KKH598/proj/state_lattice_planner/logs/sl_2.npy")

    planner = StateLatticePlanner(sl, world_fn, dt)
    planner.plan(np.array([0.0, -10.0, 0.0, 0.0]), 0.5, 0.5)

    iter = 0
    vis.add("world", world)
    vis.show()
    while vis.shown():
        iter += 1
        try:
            planner.next()
            robot_model.setConfig(planner.robot_model.getConfig())
            wheelchair_model.setConfig(planner.wheelchair_model.getConfig())
            time.sleep(dt)
        except StopIteration:
            print("Stopped at iteration ", iter)
            break
