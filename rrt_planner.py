import json
import heapq
import math
from typing import Dict, List, Tuple
import klampt
from klampt.model import collide
from klampt.math import vectorops as vo
from klampt.math import se3, so3, so2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.lib.function_base import disp

from consts import SETTINGS_PATH

import random
import copy

class RRTPlanner:
    class Node:
        def __init__(self, x, y, yaw):
            self.x = x
            self.y = y
            self.yaw = yaw
            self.v = 0
            self.dyaw = 0
            self.path_x = []
            self.path_y = []
            self.path: np.array([List[float,float,float]]) = [] #x,y,yaw
            self.parent = None

    def __init__(self, world_fn: str, target_cfg: List[float], disp_tol, rot_tol, steer_lim = 1.5, max_itr = 10000, map_range = [-20,20,-20,20], dt = 1 / 50, exp_dist = 1):
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.disp_tol = disp_tol
        self.rot_tol = rot_tol
        # Define World, Robot, Wheelchair, Model
        self.world_model = klampt.WorldModel()
        self.world_model.loadFile(world_fn)
        self.wheelchair_model: klampt.RobotModel = self.world_model.robot("wheelchair")
        self.left_handle_name = "lefthandle_link"
        self.right_handle_name = "righthandle_link"
        self.w_base_name = "base_link"
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]
        self.target_cfg = self.wheelchair_model.getConfig()
        cfg = self.wheelchair_model.getConfig()
        cfg[0] = 1
        self.wheelchair_model.getConfig()
        self.target_cfg[self.wheelchair_dofs[0]]=target_cfg[0]
        self.target_cfg[self.wheelchair_dofs[1]]=target_cfg[1]
        self.target_cfg[self.wheelchair_dofs[2]]=target_cfg[2]
        self.w_t_bl = se3.mul(
            se3.inv(
                self.wheelchair_model.link(self.w_base_name).getTransform()),
            self.wheelchair_model.link(self.left_handle_name).getTransform())
        self.w_t_br = se3.mul(
            se3.inv(
                self.wheelchair_model.link(self.w_base_name).getTransform()),
            self.wheelchair_model.link(self.right_handle_name).getTransform())

        # Set Collision 
        self.collider = collide.WorldCollider(self.world_model)
        self._ignore_collision_pairs()
        self._set_collision_margins()

        # target
        self.target_pos = np.array([
            target_cfg[0],
            target_cfg[1]
        ])
        self.target_yaw = target_cfg[2]

        # constraint
        self.EE_vel_lim = np.array(self.settings["limb_EE_velocity_limits"])
        self.base_v_lim = np.array(self.settings["base_velocity_limits"])


        q = self.wheelchair_model.link(self.w_base_name).getTransform()
        init_w_pos = q[1]
        self.start = self.Node(init_w_pos[0],init_w_pos[1],init_w_pos[2])

        self.end = self.Node(self.target_pos[0],self.target_pos[1],self.target_yaw)
        self.max_iter = max_itr
        self.node_list = []
        self.map_min_x = map_range[0]
        self.map_max_x = map_range[1]
        self.map_min_y = map_range[2]
        self.map_max_y = map_range[3]
        self.min_steer = -steer_lim 
        self.max_steer = steer_lim
        self.dt = dt
        self.exp_dist = exp_dist

        self.res = 0.5

        # self.cfree_space = self._cfree()
         
    def planning(self):

        self.node_list = [self.start]
        print("RRT START")
        for i in range(self.max_iter):
            # print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd_node = self._sample()
            nearest_ind = self._get_nearest(self.node_list, rnd_node)
            new_node = self._steer(self.node_list[nearest_ind],rnd_node)
# 
            # if i % 200 == 0:
            #     self.draw_graph(rnd_node)
                
            if new_node != None :
                #print(self._check_node_inside_map(new_node),self._collides(new_node))
                if self._check_node_inside_map(new_node) and not self._collides(new_node):
                    self.node_list.append(new_node)
                
            if self._get_dist(self.node_list[-1], self.end) <= self.disp_tol:
                # if abs(so2.diff(self.node_list[-1].yaw, self.end.yaw)) <= self.rot_tol:
                self._connect()
                return self._final_path(len(self.node_list) - 1)

        return None
        
    def _cfree(self):
        collides = True
        cfree = []
        
        for i in linspace(self.map_min_x,self.map_max_x, int(abs(self.map_max_x-self.map_min_x)/self.res)):
            for j in linspace(self.map_min_y,self.map_max_y, int(abs(self.map_max_y-self.map_min_y)/self.res)):
                for k in linspace(-math.pi,math.pi,10):
                    if not self._collides(self.Node(i,j,k)):
                        cfree.append([i,j,k])
                        
        print("Generate C free space")
        return cfree

    def _sample(self):
        
        l = int(abs(self.map_max_x-self.map_min_x)/self.res)+1

        if random.randint(0,9)>2: # sampling rate = 0.7
            rnd_pos=([random.randint(0,l)*self.res+self.map_min_x , random.randint(0,l)*self.res+self.map_min_y, random.uniform(-math.pi, math.pi)])
            # rnd_pos = self.cfree_space[random.randint(0,len(self.cfree_space)-1)]
            rnd_node = self.Node(rnd_pos[0], rnd_pos[1], rnd_pos[2])
        else:
            rnd_node = self.Node(self.target_pos[0],self.target_pos[1], self.target_yaw)

        return rnd_node
            
    def _get_nearest(self, node_list, rnd_node):
        
        dlist = [self._get_dist (node, rnd_node)
              for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def _steer(self, from_node, to_node):
        T = random.randint(5,50)*self.dt #0.1~1sec
        new_node_list = []

        for i in range(3):
            vr = random.randint(0,self.EE_vel_lim[0]*10)*0.1
            vl = random.randint(0,self.EE_vel_lim[1]*10)*0.1 
            v = (vr+vl)/2
            dyaw = (vr-vl)/0.68 #TODO: change to real wb value in CAD
            if abs(dyaw)> self.base_v_lim[2]:
                break
            new_node = None
            new_node = self.Node(from_node.x,from_node.y,from_node.yaw)
            # new_node.path.append([from_node.x,from_node.y,from_node.yaw])
            # new_node.path_x.append(from_node.x)
            # new_node.path_y.append(from_node.y)
            for _ in range (int(T/self.dt)):
                new_node.x += v * math.cos(new_node.yaw) * self.dt
                new_node.y += v * math.sin(new_node.yaw) * self.dt
                new_node.yaw += dyaw * self.dt
                new_node.path.append([new_node.x,new_node.y, new_node.yaw])
                new_node.path_x.append(new_node.x)
                new_node.path_y.append(new_node.y)
                # new_node.cost += self.get_dist(from_node, new_node)
                if self._get_dist(from_node,new_node)>=self.exp_dist:
                    break
            new_node.parent = from_node
            new_node_list.append(new_node)

        if len(new_node_list)==0:
            return None
        best_node = new_node_list[self._get_nearest(new_node_list,to_node)]
        return best_node
    
    def _final_path(self, goal_ind):
        path = []
        node = self.node_list[goal_ind]
        while node.parent is not None:
            # path.insert(0,[node.x, node.y, node.yaw])
            for i in range(len(node.path)):
                path.insert(0,[node.path[len(node.path)-i-1][0], node.path[len(node.path)-i-1][1], node.path[len(node.path)-i-1][2]])
            node = node.parent
        path.insert(0,[node.x, node.y, node.yaw])
        return path

    def _check_node_inside_map(self, node) -> bool:
        if node.x<self.map_min_x or node.x>self.map_max_x or node.y<self.map_min_y or node.y>self.map_max_y:
            return False
        return True

    def _get_dist(self, from_node, to_node) -> float:
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy) #TODO Geodesic distance
        return d

    def _collides(self, node) -> bool:
        cfg = self.wheelchair_model.getConfig()
        new_cfg = cfg[:]
        new_cfg[self.wheelchair_dofs[0]] = node.x-1
        new_cfg[self.wheelchair_dofs[1]] = node.y
        new_cfg[self.wheelchair_dofs[2]] = node.yaw
        self.wheelchair_model.setConfig(new_cfg)

        collides = False
        for _ in self.collider.collisions():
            collides = True
            break
        return collides

    def _ignore_collision_pairs(self):
        for i in range(self.wheelchair_model.numLinks()):
            for j in range(self.wheelchair_model.numLinks()):
                if i != j:
                    link_a = self.wheelchair_model.link(i)
                    link_b = self.wheelchair_model.link(j)
                    if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                        self.collider.ignoreCollision((link_a, link_b))
        # For this planner, ignore collisions between the wheelchair and
        # trina, just want to check against obstacles
        trina_model: klampt.RobotModel = self.world_model.robot("trina")
        for i in range(self.wheelchair_model.numLinks()):
            for j in range(trina_model.numLinks()):
                link_a = self.wheelchair_model.link(i)
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
        for i in range(self.wheelchair_model.numLinks()):
            link = self.wheelchair_model.link(i)
            if not link.geometry().empty():
                link.geometry().setCollisionMargin(0.75)

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y,"-g")
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([self.map_min_x, self.map_max_x,self.map_min_y, self.map_max_y])
        plt.grid(True)
        plt.pause(0.1)

    def _connect(self):
        

        from_node = self.node_list[-1]
        to_node = self.end
        previous_node = from_node
        # print(self.end.yaw)
        yaw_diff = so2.diff(self.end.yaw,from_node.yaw)
        
        if abs(yaw_diff) < self.rot_tol:
            print(abs(yaw_diff))
            pass

        n = int(abs(yaw_diff)/(self.base_v_lim[2]*self.dt))

        dyaw = so2.diff(self.end.yaw,from_node.yaw)/n

        for i in range(n):
            new_node = self.Node(previous_node.x,previous_node.y,previous_node.yaw)
            new_node.yaw += dyaw
            new_node.path.append([new_node.x,new_node.y, new_node.yaw])
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
            new_node.parent = previous_node
            self.node_list.append(new_node)
            previous_node = new_node

            # if abs(so2.diff(self.end.yaw,new_node.yaw)) < self.rot_tol:
            #     break

def main():
    import time
    from klampt import vis
    world_fn = "Model/worlds/world_short_turn.xml"
    world = klampt.WorldModel()
    world.loadFile(world_fn)
    rrt = RRTPlanner (world_fn, np.array([0.0, -10.0, 0.0]),0.5,0.5)
    start = time.monotonic()
    path = rrt.planning()
    print(time.monotonic()-start, "sec")
    if path != None:
        with open("path.json", "w") as f:
            json.dump(path, f)
        print("Find Path!!")
    else:
        print("No Path!!!")
if __name__ == "__main__":
    main()
