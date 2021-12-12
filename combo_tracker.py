import math
import cvxpy as cp
from typing import Dict, List
from klampt.math import se3, so3
from klampt.model import ik

import numpy as np
from tracker import Tracker


class ComboTracker(Tracker):
    def __init__(self, world_model, dt: float, lam: Dict[str, float]=None):
        super().__init__(world_model, dt, lam)
        self.q_dot_combo_var = cp.Variable(self.num_total_dofs, name="qdot")
        self.combo_obj = cp.Minimize(
            cp.norm2(
                self.jac_param @ self.q_dot_combo_var
                    - self.ls_target_twist_param) ** 2
            # Arm dofs
            + self.arm_penalty * cp.norm2(self.q_dot_combo_var[:self.num_arm_dofs])**2
            # Strafe dof
            + self.strafe_penalty * self.q_dot_combo_var[self.num_arm_dofs + 1]**2
            # Base dofs
            + self.base_penalty * cp.norm2(self.q_dot_combo_var[self.num_arm_dofs:])**2
            # Attractor configs
            + self.attractor_penalty * cp.norm2(self.q_dot_combo_var[:self.num_arm_dofs] * self.dt + self.arms_config_param - self.arms_attractor)**2
        )
        self.combo_constraints = self.build_constraints(self.q_dot_combo_var)
        self.combo_prob = cp.Problem(self.combo_obj, self.combo_constraints)

    def get_target_config(self, vel: np.ndarray) -> str:
        w_t_wb = self.wheelchair_model.link(self.w_base_name).getTransform()
        yaw = math.atan2(w_t_wb[0][1], w_t_wb[0][0])
        # In world frame, translation vectors from wheelchair base to handles
        w_p_w_bl = so3.apply(w_t_wb[0], self.w_t_bl[1])
        w_p_w_br = so3.apply(w_t_wb[0], self.w_t_br[1])
        omega = np.array([0, 0, vel[1]])
        vel_l = (np.array([np.cos(yaw) * vel[0], np.sin(yaw) * vel[0], 0])
            + np.cross(omega, w_p_w_bl))
        vel_r = (np.array([np.cos(yaw) * vel[0], np.sin(yaw) * vel[0], 0])
            + np.cross(omega, w_p_w_br))
        target = np.concatenate((
            omega, vel_l, omega, vel_r
        ))

        cfg = self.robot_model.getConfig()
        q_dot = self.get_q_dot(cfg, target)
        delta_q = self.dt * q_dot
        new_t_cfg = self.extract_cfg(cfg) + delta_q
        new_cfg = cfg[:]
        self.pack_cfg(new_cfg, new_t_cfg)
        self.robot_model.setConfig(new_cfg)

        # Constrain the left hand to have the same z, roll, and pitch as it
        # did initially in the world frame.
        t_wl = list(self.robot_model.link(self.left_name).getTransform())
        global_constr_to_target = so3.mul(t_wl[0],
            so3.inv(self.init_t_wl[0]))
        global_constr_to_target_rpy = list(so3.rpy(global_constr_to_target))
        # Zero out the global roll and pitch of the left hand
        for i in range(2):
            global_constr_to_target_rpy[i] = 0
        t_wl[0] = so3.mul(so3.from_rpy(global_constr_to_target_rpy),
            self.init_t_wl[0])
        # Constrain z
        p_wl = list(t_wl[1])
        p_wl[2] = self.init_t_wl[1][2]
        t_wl[1] = p_wl

        # Modify t_wl so that the wheelchair obeys differential constraints
        init_t_ww = self.wheelchair_model.link(self.w_base_name).getTransform()
        init_yaw = yaw = math.atan2(init_t_ww[0][1], init_t_ww[0][0])
        init_w_np = np.array([init_t_ww[1][0], init_t_ww[1][1], init_yaw])
        t_ww = se3.mul(se3.mul(t_wl, se3.inv(self.t_hee)), se3.inv(self.w_t_bl))
        yaw = math.atan2(t_ww[0][1], t_ww[0][0])
        w_np = np.array([t_ww[1][0], t_ww[1][1], yaw])
        d = w_np[:2] - init_w_np[:2]
        dir = np.array([np.cos(init_yaw), np.sin(init_yaw)])
        # Project the delta onto the wheelchair's forward direction
        w_delta = (d @ dir) * dir
        w_np[:2] = init_w_np[:2] + w_delta
        w_cfg = self.wu.rcfg_to_cfg(w_np)
        self.wheelchair_model.setConfig(w_cfg)

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
            return "collision"
        return "success"

    def get_q_dot(self, cfg: List[float], target: np.ndarray) -> np.ndarray:
        """Get joint velocities so that the robot's hands achieve the target
        twists in `target` while satisfying secondary objectives as well.

        Args:
            cfg (List[float]): Current full Klampt config of robot.
            target (np.ndarray): (12,) array left target twist followed by
                right target twist.

        Returns:
            np.ndarray: (self.num_total_dofs,) q dot
        """
        self.robot_model.setConfig(cfg)
        q_dot, _ = self.get_combo_soln(cfg, target)
        return q_dot

    def get_combo_soln(self, cfg: List[float], target: np.ndarray):
        self.fill_combo_params(cfg, target)
        res = self.combo_prob.solve()
        return self.q_dot_combo_var.value, res

    def fill_combo_params(self, cfg: List[float], target: np.ndarray):
        self.fill_ls_params(cfg, target)
        np_cfg = np.array(cfg)
        self.arms_config_param.value = np.concatenate((
            np_cfg[self.left_dofs], np_cfg[self.right_dofs]))
