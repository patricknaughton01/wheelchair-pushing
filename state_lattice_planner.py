from typing_extensions import Concatenate
import numpy as np
from typing import List, Dict, Tuple
from utils_.state_lattice import *
from utils_.helper import *
from utils_.tw import *
import heapq
import json
from consts import SETTINGS_PATH
from planner import Planner
import klampt
from klampt.math import se3, so3, so2
from klampt.model import ik, collide
from grid_planner import GridPlanner
from tracker import mirror_arm_config

# load state lattice logs
cfg = Config("params/tw.yaml")
sys = TWSys(cfg.value, seed = 0)
sl = StateLattice(sys)
sl.load("logs/sl.npy")

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
        print("Init SL Planner")

    def plan(self, tgt: np.ndarray, disp_tol: float, rot_tol: float):
        """get the planned trajectory with SL

        Args:
            tgt (List): target

        Returns:
            list: [trajectory data]
        """
        self.tgt = np.append(tgt, tgt[2]) # set target robot psi the same as the wheelchair
        print(f"start plan for target: {self.tgt }")
        self.tgt_idx = self._pos_to_ind(self.tgt)
        super().plan(tgt, disp_tol, rot_tol)
        cfg = self._wheelchair_np_to_cfg(self.tgt)
        # warm start the grid planner
        # gp = GridPlanner(self.world_fn, cfg, self.sl.r)
        # gp.get_dist(tgt)
        # print("warm start initialization")
        if self._collides_w(self.tgt):
            print("Tgt collides with obstacles!!!")
            self.close_set[self.tgt_idx] = float('inf')
        while (self.tgt_idx not in self.close_set) and len(self.open_set) > 0:
            _, min_dist, min_ind = heapq.heappop(self.open_set)
            if len(self.open_set)%5000 == 1:
                print(f"len open set: { len(self.open_set)}, min_ind: {min_ind}")
            if min_ind not in self.close_set:
                # print("Expanding", min_ind)
                self.close_set[min_ind] = min_dist
                suc_nodes, suc_data, orien_idx = self.get_successor(min_ind)
                for i, n_ind in enumerate(suc_nodes):
                    if n_ind in self.close_set:
                        continue
                    # print(f"suc_data['states_r'][i][-1,:]: {suc_data['states_r'][i][-1,:]}")
                    cand_cost = suc_data['costs'][i] + min_dist + self._cost_obs(n_ind, min_ind, suc_data['states_r'][i][-1,:])
                    best_known_dist = self.open_d_map.get(n_ind, float('inf'))
                    if cand_cost < best_known_dist:
                        # euclidean heuristic
                        heapq.heappush(self.open_set,(self.euclidean_heuristic(n_ind) + cand_cost, cand_cost, n_ind))
                        # Dijkstra heuristic
                        # n_pos = self._ind_to_pos(n_ind)
                        # heapq.heappush(self.open_set,(gp.get_dist(n_pos) + cand_cost, cand_cost, n_ind))
                        self.open_d_map[n_ind] = cand_cost
                        self.traj[n_ind] = {'prev': min_ind, 'state_w': suc_data['states_w'][i], 'state_r':suc_data['states_r'][i], 'action_taken': [orien_idx, i]}
        
        self.set_configs(self.init_configs) # reset robot and wheelchair pose to the initial ones after plan
        return self.retrive_traj()

    def next(self):
        # Check for termination:
        wheelchair_cfg = self.wheelchair_model.getConfig()
        wheelchair_xy = np.array([
            wheelchair_cfg[self.wheelchair_dofs[0]],
            wheelchair_cfg[self.wheelchair_dofs[1]]
        ])
        wheelchair_yaw = wheelchair_cfg[self.wheelchair_dofs[2]]
        if np.linalg.norm(wheelchair_xy - self.target[:2]) <= self.disp_tol:
            if abs(so2.diff(wheelchair_yaw, self.target[2])) <= self.rot_tol:
                print("Arrived!")
                raise StopIteration
        # gen config
        if self.cfg_ind >= self.traj_w.shape[0]:
            raise StopIteration
        res = self.get_target_config(self.traj_w[self.cfg_ind, :], self.traj_r[self.cfg_ind, :])
        if res != "success":
            print(f"fail to get target config due to {res}")
            raise StopIteration
        self.cfg_ind += 1
        return self.get_configs()
        
    def get_target_config(self, w_np, r_np) -> List[float]:
        # print(f"gen config from w_np: {w_np} and r_np: {r_np}")
        w_q = self._wheelchair_np_to_cfg(w_np)
        self.wheelchair_model.setConfig(w_q)

        r_q = self._robot_np_to_cfg(r_np)
        self.robot_model.setConfig(r_q)
        # print("configs")
        # print(np.asarray(self.wheelchair_model.getConfig())[[0,1,3]])
        # print(np.asarray(self.robot_model.getConfig())[[0,1,3]])

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
            # print(f"t_wee: {t_wee}")
            goal = ik.objective(link, R=t_wee[0], t=t_wee[1])
            if not ik.solve(goal, activeDofs=dofs):
                return "ik"
        # TODO: check what's caused the collision
        # collides = False
        # for _ in self.collider.collisions():
        #     collides = True
        #     break
        # if collides:
        #     print("collision detected!")
        #     return "collision"
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
        print("retrive trajectory")
        end = self.tgt_idx
        nodes, states_w, states_r, action_taken = [self.tgt], [], [], []
        while end != self.start_idx:
            prev = self.traj[end]['prev']
            prev_w_pose = self._ind_to_pos(prev)
            nodes.insert(0,list(prev_w_pose))
            states_w.insert(0,self.pos_l2g(prev_w_pose, self.traj[end]['state_w']))
            states_r.insert(0,self.pos_l2g(prev_w_pose, self.traj[end]['state_r']))
            action_taken.insert(0,self.traj[end]['action_taken'])
            end = prev
        print(f"nodes count: {len(nodes)}")
        self.traj_w = np.asarray(states_w).reshape(-1, 3)
        self.traj_r = np.asarray(states_r).reshape(-1, 3)
        return np.asarray(nodes), self.traj_w, self.traj_r, np.asarray(action_taken)

    # def _cost_obs(self, ind_w, ind_w_prev, pos_r_l) -> float:
    #     # TODO: can't check both somehow
    #     # pos_w = self._ind_to_pos(ind_w)
    #     # c_w = self._collides_w(pos_w)
    #     # if c_w:
    #     #     return float('inf')
    #     pos_w_prev = self._ind_to_pos(ind_w_prev)
    #     pos_r = [pos_w_prev[0] + pos_r_l[0], pos_w_prev[1] + pos_r_l[1], pos_r_l[2]]
    #     c_r = self._collides_r(pos_r)
    #     if c_r:
    #         return float('inf')
    #     return 0

    def _cost_obs(self, ind_w, ind_w_prev, pos_r_l) -> float:
        pos_w = self._ind_to_pos(ind_w)
        pos_w_prev = self._ind_to_pos(ind_w_prev)
        pos_r = [pos_w_prev[0] + pos_r_l[0], pos_w_prev[1] + pos_r_l[1], pos_r_l[2]]
        cfg_w,  cfg_r = self._wheelchair_np_to_cfg(pos_w), self._robot_np_to_cfg(pos_r)
        self.set_configs([cfg_r, cfg_w])

        collides = False
        for _ in self.collider.collisions():
            collides = True
            break
        if collides:
            return float('inf')
        return 0

    def pos_l2g(self, origin, state):
        """shift state with respect to the origin

        Args:
            origin (list): new origin
            state (ndarray): state to transform

        Returns:
            ndarray: transformed states
        """
        xy = state[:,0:2] + origin[0:2]
        psi = np.array(state[:,2]).reshape(-1,1)
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
    
    def _collides_w(self, pos_w) -> bool:
        self.wheelchair_model.setConfig(self._wheelchair_np_to_cfg(pos_w))
        collides = False
        for _ in self.collider.collisions():
            collides = True
            # print(f"collision detected on wheelchair pos: {pos_w}")
            break
        return collides

    def _collides_r(self, pos_r) -> bool:
        self.robot_model.setConfig(self._robot_np_to_cfg(pos_r))
        collides = False
        for _ in self.collider.collisions():
            collides = True
            # print(f"collision detected on robot pos: {pos_r}")
            break
        return collides
    
    def _ignore_collision_pairs(self):
        # ignore collisions between the wheelchair and robot
        for i in range(self.wheelchair_model.numLinks()):
            for j in range(self.robot_model.numLinks()):
                link_a = self.wheelchair_model.link(i)
                link_b = self.robot_model.link(j)
                if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                    self.collider.ignoreCollision((link_a, link_b))

        # only ignore gripper and handle
        # gripper_dofs = self.left_gripper_dofs + self.right_gripper_dofs
        # for name in [self.left_handle_name, self.right_handle_name]:
        #     for d in gripper_dofs:
        #         self.collider.ignoreCollision((
        #             self.wheelchair_model.link(name), self.robot_model.link(d)
        #         ))

        # Ignore TRINA self collisions
        for i in range(self.robot_model.numLinks()):
            for j in range(self.robot_model.numLinks()):
                if i != j:
                    link_a = self.robot_model.link(i)
                    link_b = self.robot_model.link(j)
                    if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                        self.collider.ignoreCollision((link_a, link_b))

        # Ignore wheelchair self collisions
        for i in range(self.wheelchair_model.numLinks()):
            for j in range(self.wheelchair_model.numLinks()):
                if i != j:
                    link_a = self.wheelchair_model.link(i)
                    link_b = self.wheelchair_model.link(j)
                    if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                        self.collider.ignoreCollision((link_a, link_b))

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
            cfg[d] = w_np[i]  + self.T_ww_init[1][i] 
        return cfg

    def _robot_np_to_cfg(self, r_np: List[float]) -> List[float]:
        cfg = self.robot_model.getConfig()
        for i, d in enumerate(self.base_dofs):
            cfg[d] = r_np[i] + self.T_rw_init[1][i]   # need to consider the initial pose of the robot
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
        # print("init configs")
        # print(np.asarray(self.wheelchair_model.getConfig())[[0,1,3]])
        # print(np.asarray(self.robot_model.getConfig())[[0,1,3]])
        goal = ik.objective(self.robot_model.link(self.left_name),
            R=t_wee[0], t=t_wee[1])
        if not ik.solve(goal, activeDofs=self.left_dofs):
            raise RuntimeError("Couldn't find initial grasp pose, quitting")
        l_cfg = self.get_arm_cfg('l', self.robot_model.getConfig()).tolist()
        r_cfg = mirror_arm_config(l_cfg)
        self.set_arm_cfg('l', l_cfg)
        self.set_arm_cfg('r', r_cfg)

        self.T_ww_init = se3.inv(self.wheelchair_model.link(self.w_base_name).getTransform())
        self.T_rw_init = se3.inv(self.robot_model.link(self.base_name).getTransform())
        print(f"w world trans: {self.T_ww_init}")
        print(f"r world trans: {self.T_rw_init}")
        self.init_configs = self.get_configs()
        print(f"cache init configs: {self.init_configs}")

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
    planner = StateLatticePlanner(sl, world_fn, dt)
    planner.plan(np.array([0, -10, 0.0]), 0.5, 0.5)

    iter = 0
    vis.add("world", world)
    vis.show()
    while vis.shown():
        iter += 1
        try:
            cfgs = planner.next()
            robot_model.setConfig(cfgs[0])
            wheelchair_model.setConfig(cfgs[1])
            time.sleep(dt)
        except StopIteration:
            print("Stopped at iteration ", iter)
            break
