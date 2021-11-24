import numpy as np
from .nmpc import *
from .trajectory import *
from typing import Dict, List, Tuple
from .helper import diff_angle

class StateLattice:
    """generate state lattice (SL) for wheelchair mode
    """
    def __init__(self, sys):
        """Init with system and optimization parameters 
        # node: (x, y, psi_w, psi_r)
        # edge: computed by solving 2 points boundary problem for 0 and pi/4 directions
        # psi idx:                          0,          1,        2,        3,   4,        5       , 6       7
        # data: rotated SL in 8 direction: [-np.pi*3/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi*3/4, np.pi]

        Args:
            sys ([System]): system and optimization parameters
        """
        self.r = sys.sets['res']
        self.sys = sys
        self.delta_psi = np.pi/4
        self.data = {}
        self.psi_set = np.linspace(-np.pi*3/4, np.pi, 8)
        self.x_curs_axis_idx = [(0,0,3,2),(0,0,3,3),(0,0,3,4)]
        self.x_curs_diag_idx = [(0,0,4,3),(0,0,4,4),(0,0,4,5)]
        self.cur_idx = [self.x_curs_axis_idx, self.x_curs_diag_idx]
        
        self.axis_idx = [[0,0,2], [0,0,4], [2, 2, 4], [2, 1, 3], [2, 1,  4], [2, 0, 4], 
                            [2, 0, 2], [2, -1, 2], [2, -1, 3], [2, -2, 2],
                            [1, 1, 5], [1, 0, 3], [1, -1, 1]]
        self.diag_idx = [[0,0,3], [0,0,5], [0, 2, 5], [0, 2, 6], [1, 2, 3], [1, 2, 4],[1, 2, 5], [2, 2, 3], 
                            [2, 2, 5], [0, 1, 6],[1, 1, 4], [2, 1, 3], [2, 1, 4], [2, 1, 5],
                           [1, 0, 2], [2, 0, 2], [2, 0, 3]]
        self.tgt_idx = [self.axis_idx, self.diag_idx]

        # actural delta pose
        # sl_axis = np.array([[2*r, 2*r, 1/4*np.pi], [2*r, r, 0], [2*r, r,  1/4*np.pi], [2*r, 0, 1/4*np.pi], 
        #                     [2*r, 0, -1/4*np.pi], [2*r, -r, -1/4*np.pi], [2*r, -r, 0], [2*r, -2*r, -1/4*np.pi],
        #                     [r, r, np.pi/2], [r, 0, 0], [r, -r, -1/2*np.pi]])
        # sl_diag = np.array([[0, 2*r, 1/2*np.pi], [0, 2*r, 3/4*np.pi], [r, 2*r, 0], [r, 2*r, 1/4*np.pi],[r, 2*r, 1/2*np.pi], [2*r, 2*r, 0], 
        #                     [2*r, 2*r, np.pi/2], [0, r, 3/4*np.pi],[r, r, 1/4*np.pi], [2*r, r, 0], [2*r, r, 1/4*np.pi], [2*r, r, 1/2*np.pi],
        #                    [r, 0, -1/4*np.pi], [2*r, 0, -1/4*np.pi], [2*r, 0, 0]])
        # self.sl_turning = np.array([[0,0,1/4*np.pi], [0,0,-1/4*np.pi]])

    def gen_data(self, sl_type:int):
        """generate state lattices database for psi = 0 and pi/4 cases
            data[node] = {'targets': xx, 'states_w': xx, 'ctrls_w': xx, 'states_r': xx, 'costs': xx}

        Args:
            sl_type (int): 0 for axis (psi = 0) case, 1 for diagonal (psi = pi/4) case

        """
        for x_cur_idx in self.cur_idx[sl_type]:
            tgts, states_w, ctrls_w, states_r, costs = [], [], [], [], []
            for x in self.tgt_idx[sl_type]:
                for i in np.linspace(-1,1,3):
                    x_tgt_idx = x + [x[2]+int(i)]
                    self.update_umin(x) # handle turning case, as the u is constrained to positive for other cases
                    opt = nmpc(self.sys)
                    success = opt.step(stateCurrent = self.idx2pos(x_cur_idx), stateTarget = self.idx2pos(x_tgt_idx), interpolation=False, initMethod = 'linear')
                    if success:
                        tgts.append(x_tgt_idx)
                        states_w.append(np.round(opt.state_sol,2))
                        ctrls_w.append(np.round(opt.ctrl_sol,2))
                        states_r.append(np.round(opt.robot_state_sol,2))
                        costs.append(np.round(opt.cost_value,2))
                        # print(opt.cost_value)
                # self.update_data(states_wi, ctrls_wi, states_ri, costs_i)
            self.data.update({x_cur_idx: {'targets': tgts, 'states_w': states_w, 'ctrls_w': ctrls_w, 'states_r': states_r, 'costs': costs}})
        print(f"generated {len(tgts)} primitives!")

    def gen_data_full(self):
        """generate the full dataset for SL
        8 directions in total, the first two are from solving optimization problems 
        and the rest are from translating the first two sets
        """
        self.gen_data(0)
        self.gen_data(1)
        self.trans_sl()
        self.keys = list(self.data.keys())

    def trans_sl(self):
        """Transform SL 
        sl_axis (sa, psi_w = 0 case) and sl_diag (sd, psi_w = pi/4 case) are generated from solving optimization problem
        the other 6 direction of SL are generated from these two
        idx:   
        0       sd.dot(R(pi))
        1       sa.dot(R(-pi/2))
        2       sd.dot(R(-pi/2))
        3       sa.dot(R(0))
        4       sd.dot(R(0))
        5       sa.dot(R(pi/2))
        6       sd.dot(R(pi/2))
        7       sa.dot(R(pi))
        """
        for psi_idx_inc in range(8): # theta is delta idx for psi 
            if psi_idx_inc in [0,1]: # generated from solving optimization problems
                continue
            # print(f"working on delta theta {psi_idx_inc}*pi/4")
            sl_type = 0 if psi_idx_inc%2 == 0 else 1
            psi_idx_inc -= sl_type
            for x_cur_idx in self.cur_idx[sl_type]:
                psi_w_n = (x_cur_idx[2] + psi_idx_inc) % 8
                psi_r_n = (x_cur_idx[3] + psi_idx_inc) % 8
                x_cur_new = (0,0,psi_w_n,psi_r_n)
                # print(f"x_cur_idx: {x_cur_idx}")
                # print(f"x_cur_new: {x_cur_new}")
                self.data[x_cur_new] = self.trans_data(self.data[x_cur_idx], psi_idx_inc)
        
    def trans_data(self, data, psi_idx_inc):
        """Transform SL data based to certain orientation

        Args:
            data ([dict]): [intial SL from solving optimization problems]
            psi_idx_inc ([int]): [indx of delta theta (pi/4)]

        Returns:
            [dict]: [transfromed SL in given orientation]
        """
        n = len(data['states_w'])
        states_w, states_r = [], []
        theta = self.psi_set[(psi_idx_inc + 3)%8] 
        R = self.M_R(theta)

        tgts = self.trans_state_idx(data['targets'], R, psi_idx_inc)

        for i in range(n):
            states_w.append(self.trans_state(data['states_w'][i], R, psi_idx_inc))
            states_r.append(self.trans_state(data['states_r'][i], R, psi_idx_inc))
        return {'targets': tgts, 'states_w': states_w, 'ctrls_w': data['ctrls_w'], 'states_r': states_r, 'costs': data['costs']}

    def trans_state(self, state, R, psi_idx_inc):
        """translate state with given delta_psi

        Args:
            state ([list/array]): [state: [x,y,psi_w,psi_r]]
            R (ndarray): rotation matrix
            psi_idx_inc (int): index for delta psi 

        Returns:
            ndarray: transformed state
        """
        state = np.asarray(state)
        xy = self.trans_xy(state[:, 0:2], R)
        psi = self.trans_psi(state[:, 2:], psi_idx_inc)
        return np.concatenate((xy,psi), axis = 1)

    def trans_state_idx(self, state, R, psi_idx_inc):
        """translate state with given delta_psi

        Args:
            state ([list/array]): [state: [x,y,psi_w,psi_r]]
            R (ndarray): rotation matrix
            psi_idx_inc (int): index for delta psi 

        Returns:
            ndarray: transformed state
        """
        state = np.asarray(state)
        xy = self.trans_xy(state[:, 0:2], R)
        psi = (state[:, 2:] + psi_idx_inc) % 8
        return np.concatenate((xy,psi), axis = 1).astype(int)

    def trans_xy(self, xy, R):
        """translate xy coordinates

        Args:
            xy (ndarray/list): xy coordinates
            R (dnarray): rotation matrix

        Returns:
            [ndarray]: [rotated xy]
        """
        return np.round(np.dot(R, xy.T).T,2)

    def trans_psi(self, psi, psi_idx_inc):
        """transform psi

        Args:
            psi (list): psi
            psi_idx_inc (int): index for delta psi 

        Returns:
            [ndarray]: transformed psi
        """
        # return (np.asarray(psi) + psi_idx_inc) % 8
        delta_psi = -psi_idx_inc * np.pi/4
        for i in range(psi.shape[0]):
            for j in range(psi.shape[1]):
                psi[i,j] = diff_angle(psi[i,j], delta_psi)
        return psi

    def idx2pos(self, idx: Tuple[int, int, int, int]) -> np.ndarray:
        """index to position

        Args:
            idx (Tuple[int, int, int, int]): node idx

        Returns:
            np.ndarray: node pose
        """
        # print(f"{type(idx)}: {idx}")
        return np.array([idx[0]*self.r, idx[1]*self.r, self.psi_set[idx[2]], self.psi_set[idx[3]]])
    
    def M_R(self, psi):
        """compute rotation matrix for SL

        Args:
            psi ([float]): [rotation angle]

        Returns:
            [nparray]: [rotation matrix]
        """
        return np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])

    def update_umin(self, x):
        """change the control min constraint for adding pure rotation movement
        as negative control will trigger bakward motion

        Args:
            x (list): xy state, [0,0] means pure rotation
        """
        if x[0] == 0 and x[1] == 0:
            self.sys.sets['u_min'] = [-i for i in self.sys.sets['u_max']]
        else:
            self.sys.sets['u_min'] = [0, 0]

    def save(self, path):
        logs = {}
        logs['data'] = self.data
        np.save(path,logs)
        
    def load(self, path):
        logs = np.load(path, allow_pickle = True)
        items = logs.item()
        self.data = items['data']
        self.keys = list(self.data.keys())
        print("Load state lattice")

    def plot_SL(self, axs, w_orien_idx = 0,  r_orien_idx= 0, plot_dir = False, label = False, plot_cost=False, plot_idx = False):
        """plot SL

        Args:
            axs (handle)
            w_orien_idx (int, optional): select form [0 to 7]. Defaults to 0.
            r_orien_idx (int, optional): select from [0 to 2]. Defaults to 0.
            plot_dir (bool, optional): [description]. Defaults to False.
            label (bool, optional): [description]. Defaults to False.
            plot_cost (bool, optional): [description]. Defaults to False.
            plot_idx (bool, optional): [description]. Defaults to False.
        """
        data = self.data[self.keys[w_orien_idx*3 + r_orien_idx]]
        n = len(data['states_w'])
        label_idx = data['states_w'][0].shape[0]//4

        for i in range(n):
            axs.plot(data['states_w'][i][:,0],data['states_w'][i][:,1], linestyle = '-', marker = '', color = 'blue')  
            axs.plot(data['states_r'][i][:,0],data['states_r'][i][:,1], linestyle = '--', marker = '', color = "orange") 
            axs.plot(data['states_w'][i][0,0],data['states_w'][i][0,1], marker = '>', color = 'darkblue')  
            axs.plot(data['states_r'][i][0,0],data['states_r'][i][0,1], marker = '>', color = 'red') 
            axs.plot(data['states_w'][i][-1,0],data['states_w'][i][-1,1], marker = 'o', color = 'g')
            if plot_idx:
                axs.text(data['states_w'][i][label_idx,0], data['states_w'][i][label_idx,1], i)#bbox=dict(fill=False, edgecolor='red', linewidth=2)
            if plot_cost:
                axs.text(data['states_w'][i][label_idx,0], data['states_w'][i][label_idx,1]+0.1*(r_orien_idx-1), "{:.2f}".format(data['costs'][i]))#bbox=dict(fill=False, edgecolor='red', linewidth=2)
            if plot_dir:
                # if np.tan(data['states_r'][i].obs[-1,2])<5 and np.tan(data['states_r'][i][-1,2])>-5:
                    # axs.arrow(data['states_r'][i][-1,0], data['states_r'][i][-1,1], 0.1, np.tan(data['states_r'][i][-1,2])*0.1, head_width=0.05, head_length=0.05 , fc='yellow', ec='grey') 
                if np.tan(data['states_w'][i].obs[-1,2])<5 and np.tan(data['states_w'][i][-1,2])>-5:
                    axs.arrow(data['states_w'][i][-1,0], data['states_w'][i][-1,1], 0.1, np.tan(data['states_w'][i][-1,2])*0.1, head_width=0.05, head_length=0.05 , fc='blue', ec='grey') 
        # axs.arrow(0, 0, 0, 0.1, head_width=0.05, head_length=0.05 , fc='blue', ec='grey')
        if label:
            axs.plot(data['states_w'][-1][0,0], data['states_w'][-1][0,1], linestyle = '-', marker = '', color = 'blue', label = "wheelchair")  
            axs.plot(data['states_r'][-1][0,0], data['states_r'][-1][0,1], linestyle = '-', marker = '', color = "orange", label = "robot") 
            # print(f"keys: {keys}")
        axs.legend(ncol=6,loc='upper left')