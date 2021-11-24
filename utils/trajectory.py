import numpy as np
from collections import namedtuple

def zerosInitialization(obs_dim, ctrl_dim, size):
    """
    Create an all zeros trajectory.

    Parameters
    ----------
    obs_dim: int
        Observation dimension

    ctrl_dim: int
        control dimension

    size : int
        Size of trajectory
    """
    obs = np.zeros((size, obs_dim))
    ctrls = np.zeros((size, ctrl_dim))
    return Trajectory(size, obs, ctrls)
    
class Trajectory:
    """
    A discrete-time state and control trajectory.
    """
    def __init__(self, size, obs, ctrls):
        """
        Parameters
        ----------
        size : int
            Number of time steps in the trajectrory

        obs : numpy array of shape (size, system.obs_dim)
            Observations at all timesteps

        ctrls : numpy array of shape (size, system.ctrl_dim)
            Controls at all timesteps.
        """
        self._size = size
        self._obs = obs
        self._ctrls = ctrls

    def __eq__(self, other):
        return (self._size == other._size
                and np.array_equal(self._obs, other._obs)
                and np.array_equal(self._ctrls, other._ctrls))

    def __len__(self):
        return self._size

    @property
    def size(self):
        """
        Number of time steps in trajectory
        """
        return self._size

    @property
    def obs(self):
        """
        Get trajectory observations as a numpy array of
        shape (size, self.system.obs_dim)
        """
        return self._obs

    @obs.setter
    def obs(self, obs):
        if obs.shape != self._obs.shape:
            raise ValueError("obs is wrong shape")
        self._obs = obs[:]

    @property
    def ctrls(self):
        """
        Get trajectory controls as a numpy array of
        shape (size, self.system.ctrl_dim)
        """
        return self._ctrls

    @ctrls.setter
    def ctrls(self, ctrls):
        if ctrls.shape != self._ctrls.shape:
            raise ValueError("ctrls is wrong shape")
        self._ctrls = ctrls[:]




 

