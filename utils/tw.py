import numpy as np
from .helper import diff_angle
from .nmpc import *

class TWSys():
    def __init__(self, cfg, seed = 0):
        print("Init wheelchair Model")
        self.sets = cfg['space_sets']
        self.ctrl_dim = cfg['system']['ctrl_dim']
        self.obs_dim = cfg['system']['obs_dim']
        self.name = cfg['system']['name']
        self.optParam = cfg['trajOpt']
        self.opt = nmpc(self)

    @property
    def model(self):
        """
        dynamics in casadi format
        @return: lambda function
        """
        return lambda x, u: ca.horzcat(
            (u[0]+u[1])/2*np.cos(x[2]),
            (u[0]+u[1])/2*np.sin(x[2]),
            (u[1]-u[0])/0.6 # 2r = 0.6
        )

    def dynamics(self, x, u):
        """
        Dynamics of cartpole system
        @param y: array, states
        @param u: array, control

        @Return
            A list describing the dynamics of the cartpole
        """
        return np.array([(u[0]+u[1])/2*np.cos(x[2]),
                        (u[0]+u[1])/2*np.sin(x[2]),
                        (u[1]-u[0])/0.6])

    def next(self, y, u):
        """
        Forward simulation with dynamics 
        @param y: array, states
        @param u: array, control
        @param reverse: bool, True for viable set and False for reachable set
        @return y: array, next states given previous state and control input
        """
        y += self.dt * self.dynamics(y, u)
        y[2] = diff_angle(y[2],0)
        return y
        