import json
import heapq
import math
from typing import Dict, List, Tuple
import klampt
from klampt.model import ik, collide
from klampt.math import vectorops as vo
from klampt.math import se3, so3, so2
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pdb
from consts import SETTINGS_PATH
import time
import random
import copy

from rrt_planner import RRTPlanner

class DecoupledRRTPlanner:
    class Node:
        def __init__(self, x, y, yaw):
            self.x = x
            self.y = y
            self.yaw = yaw
            self.path=[]#: np.array([List[float,float,float]]) = [] #x,y,yaw
            self.path_x=[]
            self.path_y=[]
            self.parent = None
            self.ind = 0

    def __init__(self, world_fn: str, dt, max_itr = 20000, exp_dist = 0.01, eval_iter=10):
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.world_fn =world_fn

        # Define World, Robot, Wheelchair, Model
        self.left_name = "left_tool_link"
        self.right_name = "right_tool_link"
        self.base_name = "base_link"
        self.left_handle_name = "lefthandle_link"
        self.right_handle_name = "righthandle_link"
        self.w_base_name = "base_link"
        self.world_model = klampt.WorldModel()
        self.world_model.loadFile(world_fn)
        self.robot_model: klampt.RobotModel = self.world_model.robot("trina")
        self.wheelchair_model: klampt.RobotModel = self.world_model.robot(
            "wheelchair")
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
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]
        self.num_arm_dofs = len(self.left_dofs) + len(self.right_dofs)
        self.num_total_dofs = self.num_arm_dofs + len(self.base_dofs)
        self._init_config()
        self.t_lr = se3.mul(se3.inv(
            self.robot_model.link(self.left_name).getTransform()),
            self.robot_model.link(self.right_name).getTransform())

        # Constraint limits
        self.v_lim = np.array(self.settings["limb_velocity_limits"])
        self.base_v_lim = np.array(self.settings["base_velocity_limits"])
        self.q_upper_lim = np.array(self.settings["limb_position_upper_limits"])
        self.q_lower_lim = np.array(self.settings["limb_position_lower_limits"])
        self.full_q_upper_lim = np.concatenate((
            self.q_upper_lim, self.q_upper_lim
        ))
        self.full_q_lower_lim = np.concatenate((
            self.q_lower_lim, self.q_lower_lim
        ))

        # Set Collision 
        self.collider = collide.WorldCollider(self.world_model)
        self._ignore_collision_pairs()

        # Decoupled RRT Setting
        self.node_list = []
        self.dt = dt
        self.exp_dist = exp_dist
        self.max_iter = max_itr

        # init node
        q = self.robot_model.link(self.base_name).getTransform()
        self.start = self.Node(q[1][0], q[1][1], q[1][2])
        self.start.ind = 0

        #vis
        self.cfg_ind = 0

        #TODO reachable space
        # self.reach = self._reachable_space()

        self.path_robot = []
        self.path = None

    def plan(self, target_cfg, disp_tol, rot_tol):
        # target
        self.target_pos = np.array([
            target_cfg[0],
            target_cfg[1]
        ])
        self.target_yaw = target_cfg[2]
        self.target = self.Node(self.target_pos[0],self.target_pos[1],self.target_yaw)

        self.disp_tol = disp_tol

        rrt = RRTPlanner (self.world_fn, target_cfg, disp_tol, rot_tol)
        path = rrt.planning() 
        # self.cfree_space = rrt.cfree_space

        if path is None:
            print("RRT Fail")
            return None 
            
        self.set_configs(self.init_configs)
        # /print(self.wheelchair_model.getConfig()[0])
        self.node_list = [self.start]
        print("Decoupled RRT START")

        # Path of Wheelchair
        self.path = path
        self.w_path_x = []
        self.w_path_y = []
        for i in range(len(path)):
            self.w_path_x.append(path[i][0])
            self.w_path_y.append(path[i][1])

        self.w_ind = 1
        w_pos = self.Node(self.path[self.w_ind][0], self.path[self.w_ind][1], self.path[self.w_ind][2])
        for i in range(self.max_iter):

            if self.w_ind >= len(self.path) or self._get_dist(w_pos, self.target) <= self.disp_tol:
                self._connect()
                return self._final_path(len(self.node_list) - 1)


            # print("Iter:", i, ", number of nodes:", len(self.node_list), self.w_ind, len(self.path_robot))
            w_pos = self.Node(self.path[self.w_ind][0], self.path[self.w_ind][1], self.path[self.w_ind][2])

            
            w_rnd = w_pos
            # nearest_ind = self._get_nearest(self.node_list, w_rnd)
            new_node = self._steer(self.node_list[-1],w_rnd)  
            
            
            if new_node != None:
                self.node_list.append(new_node)
            # else:
            #     # print("None")
        
            # if i % 1000 == 0:
            #     self.draw_graph(self.node_list[-1],w_pos)

        return None

    def _reachability(self, r_pos, w_pos):

        orig_w_cfg = self.wheelchair_model.getConfig() # original wheelchair configuration
        w_q = orig_w_cfg[:] 
        w_q[self.wheelchair_dofs[0]] = w_pos.x-1
        w_q[self.wheelchair_dofs[1]] = w_pos.y
        w_q[self.wheelchair_dofs[2]] = w_pos.yaw 
        self.wheelchair_model.setConfig(w_q)

        # Update rotob base's position
        cfg = self.robot_model.getConfig()
        new_cfg = cfg[:]
        new_cfg[self.base_dofs[0]] = r_pos.x
        new_cfg[self.base_dofs[1]] = r_pos.y
        new_cfg[self.base_dofs[2]] = r_pos.yaw 
        self.robot_model.setConfig(new_cfg)

        self.r_t_w = se3.mul(
            se3.inv(
                self.robot_model.link(self.base_name).getTransform()),
            self.wheelchair_model.link(self.w_base_name).getTransform())

        if self.r_t_w[1][0]<0.7 or self.r_t_w[1][0]>1 or abs(self.r_t_w[1][1])>0.2:
            return "Reachability"
        
        return "success"
        
    def _connect(self):

        w_t_r = se3.mul(
            se3.inv(
                self.wheelchair_model.link(self.w_base_name).getTransform()),
            self.robot_model.link(self.base_name).getTransform())

        for i in range(self.w_ind, len(self.path)):
            
            w_pos = self.Node(self.path[self.w_ind][0], self.path[self.w_ind][1], self.path[self.w_ind][2])

            orig_w_cfg = self.wheelchair_model.getConfig() # original wheelchair configuration
            w_q = orig_w_cfg[:] 
            w_q[self.wheelchair_dofs[0]] = w_pos.x-1
            w_q[self.wheelchair_dofs[1]] = w_pos.y
            w_q[self.wheelchair_dofs[2]] = w_pos.yaw 
            # print(w_pos.x-1, w_pos.y)
            self.wheelchair_model.setConfig(w_q)

            r_cfg = se3.mul(self.wheelchair_model.link(self.w_base_name).getTransform(),w_t_r)

            # Update rotob base's position
            new_node = self.Node(r_cfg[1][0], r_cfg[1][1], math.atan2(r_cfg[0][1], r_cfg[0][0]))
            new_node.ind = self.w_ind
            new_node.parent = self.node_list[-1]
            self.node_list.append(new_node)
            self.w_ind += 1

    def get_target_config(self, node, w_pos) -> List[float]:
        orig_w_cfg = self.wheelchair_model.getConfig() # original wheelchair configuration
        w_q = orig_w_cfg[:] 
        w_q[self.wheelchair_dofs[0]] = w_pos.x-1
        w_q[self.wheelchair_dofs[1]] = w_pos.y
        w_q[self.wheelchair_dofs[2]] = w_pos.yaw 
        # print(w_pos.x-1, w_pos.y)
        self.wheelchair_model.setConfig(w_q)
        
        # In world frame, translation vectors from wheelchair base to handles
        w_t_wb = self.wheelchair_model.link(self.w_base_name).getTransform()

        # Update rotob base's position
        cfg = self.robot_model.getConfig()
        new_cfg = cfg[:]
        new_cfg[self.base_dofs[0]] = node.x
        new_cfg[self.base_dofs[1]] = node.y
        new_cfg[self.base_dofs[2]] = node.yaw 
        self.robot_model.setConfig(new_cfg)

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
            if not ik.solve_nearby(goal,1,activeDofs=dofs):
               return "ik"

        collides = False
        for _ in self.collider.collisions():
            collides = True
            break
        if collides: #TODO
            return "collision"

        #TODO
        return "success"

    def get_configs(self) -> Tuple[List[float], List[float]]:
        return self.robot_model.getConfig(), self.wheelchair_model.getConfig()

    def set_configs(self, cfgs: Tuple[List[float], List[float]]):
        print("set_configs")
        self.robot_model.setConfig(cfgs[0])
        self.wheelchair_model.setConfig(cfgs[1])

    def get_arms_cfg(self, cfg: List[float]) -> np.ndarray:
        return np.array(cfg)[self.left_dofs + self.right_dofs]

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

    def _init_config(self):
        """Set the robot's config to grasp the handles of the wheelchair.
        Done using IK to allow quick editing of the desired relative handle
        to hand transform without any file modification, although this could be
        streamlined to automatically save newly found grasping configs. Mirrors
        the left arm config to the right arm (assumes wheelchair is centered
        and symmetric).

        Raises:
            RuntimeError: If no IK solution is found, no initial grasp can
                be performed, throw an error.
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
        
        self.init_configs = self.get_configs()

    def _get_t_hee(self) -> tuple:
        # Hand tilted 45 degrees down toward the wheelchair handle
        aa = ((0, 1, 0), np.pi / 4)
        r_hee = so3.from_axis_angle(aa)
        return (r_hee, (-0.05, 0, 0.15))
        
    def _sample_rnd(self):
        # if random.randint(0,10)>2:
        #     path_length = len(self.path)
        #     rnd_idx = random.randint(0,path_length-1)
        #     rnd_node = self.Node(self.path[rnd_idx][0], self.path[rnd_idx][1], self.path[rnd_idx][2])
        # else:
        #     rnd_node = self.Node(self.path[-1][0], self.path[-1][1], self.path[-1][2])
        # return rnd_node
        rnd_pos = self.cfree_space[random.randint(0,len(self.cfree_space)-1)]
        rnd_node = self.Node(rnd_pos[0], rnd_pos[1], rnd_pos[2])
        return rnd_node
            
    def _get_nearest(self, node_list, rnd_node):
        dlist = [self._get_dist (node, rnd_node)
              for node in node_list]
        minind = dlist.index(min(dlist))
        
        return minind

    def _steer(self, from_node, to_node):

        new_node = self.Node(from_node.x,from_node.y,from_node.yaw)
        dist  = self._get_dist(from_node, to_node)

        new_node.path.append([new_node.x,new_node.y, new_node.yaw])
        new_node.path_x.append(new_node.x)
        new_node.path_y.append(new_node.y)  

        check = self._reachability(new_node, to_node)
        if check == "success":
            new_node.ind = self.w_ind
            new_node.parent = from_node
            self.w_ind += 1
            return new_node

        for i in range (min(2,int(dist/0.02))):
            # print(check)
            new_node.x += (0.02/dist)*(to_node.x-from_node.x)
            new_node.y += (0.02/dist)*(to_node.y-from_node.y)
            new_node.yaw += (0.02/dist)*so2.diff(to_node.yaw,from_node.yaw)
            # new_node.cost += self.get_dist(from_node, new_node)
            new_node.path.append([new_node.x,new_node.y, new_node.yaw])
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)  

            check = self._reachability(new_node, to_node)
            # print(check)
            if check == "success":
                new_node.ind = self.w_ind
                new_node.parent = from_node
                self.w_ind += 1
                return new_node

    
    def _final_path(self, goal_ind):
        print("Generate Final Path")
        path = []
        node = self.node_list[goal_ind]
        itr = 0
        # print(goal_ind)
        while node.parent is not None:
            # print(itr) #Fix here
            itr+=1
            path.insert(0,[node.x, node.y, node.yaw, node.ind])
            #for i in range(len(node.path)):
                #
            #    path.insert(0,[node.path[len(node.path)-i-1][0], node.path[len(node.path)-i-1][1], node.path[len(node.path)-i-1][2]])
            node = node.parent
        path.insert(0,[node.x, node.y, node.yaw, node.ind])
        self.path_robot = path

        print("Finish Path Generation")

    def _get_dist(self, from_node, to_node) -> float:
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        return d

    def _collides(self, w_state, r_node) -> bool:
        cfg = self.robot_model.getConfig()
        cfg[self.base_dofs[0]] = r_node.x
        cfg[self.base_dofs[1]] = r_node.y
        cfg[self.base_dofs[2]] = r_node.yaw
        self.robot_model.setConfig(cfg)
        
        wh = self.wheelchair_model.getConfig()
        wh[self.wheelchair_dofs[0]] = w_state.x-1
        wh[self.wheelchair_dofs[1]] = w_state.y
        wh[self.wheelchair_dofs[2]] = w_state.yaw
        self.wheelchair_model.setConfig(wh)        

        collides = False
        for _ in self.collider.collisions():
            collides = True
            break
        return collides

    def _ignore_collision_pairs(self):
        # Ignore collisions between hands and handles
        arm_dofs = self.left_gripper_dofs + self.right_gripper_dofs
        for name in [self.left_handle_name, self.right_handle_name]:
            for d in arm_dofs:
                self.collider.ignoreCollision((
                    self.wheelchair_model.link(name), self.robot_model.link(d)
                ))

        # Ignore TRINA self collisions
        for i in range(self.robot_model.numLinks()):
            for j in range(self.robot_model.numLinks()):
                if i != j:
                    link_a = self.robot_model.link(i)
                    link_b = self.robot_model.link(j)
                    if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                        self.collider.ignoreCollision((link_a, link_b))

        # for i in range(self.wheelchair_model.numLinks()):
        #     for j in range(self.robot_model.numLinks()):
        #         link_a = self.wheelchair_model.link(i)
        #         link_b = self.robot_model.link(j)
        #         if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
        #             self.collider.ignoreCollision((link_a, link_b))

        # Ignore any collision with the floor
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

    def draw_graph(self, rnd=None, new_node=None, w_pos = None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        
        if new_node is not None:
            plt.plot(new_node.x, new_node.y, "^r")

        if w_pos is not None:
            plt.plot(w_pos.x, w_pos.y, "^r")

        for node in self.node_list:
            if node.parent:
                plt.plot(node.x, node.y,"g")
        plt.plot(self.w_path_x, self.w_path_y,'k')
        plt.plot(self.start.x, self.start.y, "xr")
        plt.axis([-20,20,-20,20])
        # plt.axis([-5,5,-10,5])
        plt.grid(True)
        plt.pause(0.1)

    def next(self):
        # Check for termination:
        # print("next initiated")
        # print(len(self.path), len(self.path_robot))
        if self.path == None or len(self.path_robot)== 0:
            print("Fail to planning")
            raise StopIteration
        path = self.path
        path_robot = self.path_robot
        # print(self.cfg_ind)
        w_node = self.Node(path[path_robot[self.cfg_ind][3]][0],path[path_robot[self.cfg_ind][3]][1],path[path_robot[self.cfg_ind][3]][2])
        r_node = self.Node(path_robot[self.cfg_ind][0],path_robot[self.cfg_ind][1],path_robot[self.cfg_ind][2])
        res = self.get_target_config(r_node, w_node)
        self.cfg_ind =self.cfg_ind + 1
        if self.cfg_ind >= len(self.path_robot):
            raise StopIteration
        
        return self.get_configs()


def mirror_arm_config(config):
    """Mirrors an arm configuration from left to right or vice versa. """
    RConfig = []
    RConfig.append(-config[0])
    RConfig.append(-config[1]-math.pi)
    RConfig.append(-config[2])
    RConfig.append(-config[3]+math.pi)
    RConfig.append(-config[4])
    RConfig.append(-config[5])
    return RConfig    

def main():
    import time
    from klampt import vis
    world_fn = "Model/worlds/world_short_turn.xml"
    world = klampt.WorldModel()
    world.loadFile(world_fn)
    robot_model = world.robot("trina")
    wheelchair_model = world.robot("wheelchair")

    dt = 1 / 50
    check = time.monotonic()
    rrt_de = DecoupledRRTPlanner (world_fn, dt)
    rrt_de.plan(np.array([0.0, -10.0, math.pi]), 0.5, 0.5)
    iter = 0
    print(time.monotonic()-check)

    print("vis start")
    vis.add("world", world)
    vis.show()
    # print("before iter")

    while vis.shown():     
        iter += 1  
        try:
            cfgs = rrt_de.next()
            robot_model.setConfig(cfgs[0])
            wheelchair_model.setConfig(cfgs[1])
            time.sleep(dt)
        
        except StopIteration:
            print("END")
            time.sleep(20)
            break

if __name__ == "__main__":
    main()
