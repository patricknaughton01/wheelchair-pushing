import matplotlib.pyplot as plt
import numpy as np
from .trajectory import *

def plotCfg1(size = [6,6], sub = 1,  title = '', labels = [], fontsize = 14):
    """plot configuration

    Args:
        size (list, optional): figure size. Defaults to [6,6].
        sub (int, optional): number of subplots. Defaults to 1.
        title (str, optional): title. Defaults to ''.
        labels (list, optional): label for [y0, y1, ... , yn, x], number should be sub+1. Defaults to [].

    Returns:
        handle: axs
    """
    fig, axs = plt.subplots(sub,1,figsize=(size[0],size[1]))
    fig.suptitle(title, fontsize=fontsize, y = 0.92)
    if sub > 1:
        for i in range(sub):
            axs[i].set_ylabel(labels[i], fontsize = fontsize)
            axs[i].legend() #ncol=7,loc='upper left'
            axs[i].grid()
        axs[sub-1].set_xlabel(labels[sub], fontsize = fontsize)
    else:
        axs.set_ylabel(labels[0], fontsize = fontsize)
        axs.legend()  
        axs.grid()
        axs.set_xlabel(labels[1], fontsize = fontsize)
    return axs


def plot_sl_idx(sl_idx):
    axs = plotCfg1(size = [6,6], sub = 1,  title = '', labels = ['x (m)', 'y (m)'])
    axs.plot(sl_idx[:,0], sl_idx[:,1], linestyle = '', marker = 'o')  
    axs.legend()
    return axs

def plotTraj(states_w, states_r, arrow = False):
    traj_len = len(states_w)
    axs = plotCfg1(size = [6,6], sub = 1,  title = '', labels = ['x (m)', 'y (m)'])
    axs.plot(states_w[:,0], states_w[:,1], marker = '', label = 'Wheelchair')  
    axs.plot(states_r[:,0], states_r[:,1], marker = '', label = "Robot") 
    axs.plot(states_w[0,0], states_w[0,1], marker = '>', color = 'red')  
    axs.plot(states_r[0,0], states_r[0,1], marker = '>', color = 'red') 
    axs.plot(states_w[traj_len-1,0], states_w[traj_len-1,1], marker = 'o', color = 'g')  
    axs.plot(states_r[traj_len-1,0], states_r[traj_len-1,1], marker = 'o', color = 'g') 
    if arrow:
        for i in range(traj_len):
            if i%5 == 0 or i == traj_len-1:
                axs.arrow(states_r[i,0], states_r[i,1], 0.1, np.tan(states_r[i,2])*0.1, head_width=0.05, head_length=0.05 , fc='yellow', ec='grey')
                axs.arrow(states_w[i,0], states_w[i,1], 0.1, np.tan(states_w[i,2])*0.1, head_width=0.05, head_length=0.05 , fc='lightskyblue', ec='grey') 
    axs.legend(ncol=6,loc='upper left')
    return axs

def plotCtrl(ctrl_w):
    traj_len = len(ctrl_w)
    t = np.linspace(0, traj_len*0.02, traj_len)
    axs = plotCfg1(size = [6,6], sub = 1,  title = '', labels = ['u (m/s)','t (s)'])
    axs.plot(t, ctrl_w[:,0], marker = '', label = r'$v_l$')  
    axs.plot(t, ctrl_w[:,1], marker = '', label = r'$v_r$') 
    axs.legend()

def plot_path(path):
    axs = plotCfg1(size = [6,6], sub = 1,  title = '', labels = ['y (m)', 'x (m)'])
    axs.plot(path[:,0], path[:,1], linestyle = '-', marker = 'o')  
    axs.legend()
    return axs
