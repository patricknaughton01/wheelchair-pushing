import json
import klampt.plan as kp
from klampt.math import se3
import cvxpy as cp
import numpy as np
from consts import SETTINGS_PATH


class Tracker:
    def __init__(self, world_model, lock_arms: bool=True):
        """Create an optimization problem that generates joint motions to
        achieve a desired hand twist.

        Args:
            world_model (klampt.WorldModel): world model of klampt
        """
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.left_name = "left_tool_link"
        self.right_name = "right_tool_link"
        self.lock_arms = lock_arms
        self.world_model = world_model
        self.robot_model = self.world_model.robot(0)
        self.cspace = kp.robotcspace.RobotCSpace(self.robot_model)
        self.left_dofs = self.settings["left_arm_dofs"]
        self.right_dofs = self.settings["right_arm_dofs"]
        self.base_dofs = self.settings["base_dofs"]
        self.num_total_dofs = (
            len(self.left_dofs) + len(self.right_dofs) + len(self.base_dofs)
        )

        # Constraint limits
        self.v_lim = self.settings["limb_velocity_limits"]
        self.q_upper_lim = self.settings["limb_position_upper_limits"]
        self.q_lower_lim = self.settings["limb_position_lower_limits"]
        self.a_lim = 1


        # Unified controller optimization problems
        self.m = 6  # Dimensionality of a twist
        self.num_arms = 2
        self.num_klampt_dofs = len(self.robot_model.getConfig())
        self.ls_target_twist_param = cp.Parameter(self.m * self.num_arms, name="v_target")
        self.ls_jac_param = cp.Parameter((self.m * self.num_arms, self.num_total_dofs), name="jacobian")
        self.ls_config_param = cp.Parameter(self.num_klampt_dofs, name="config")
        self.ls_q_dot_var = cp.Variable(self.num_total_dofs,name="qdot")
        self.ls_objective = cp.Minimize(
            cp.norm2(self.ls_jac_param @ self.ls_q_dot_var - self.ls_target_twist_param)
            ** 2
        )
        self.ls_constraints = self.build_constraints(
            self.ls_q_dot_var,
            self.ls_config_param
        )
        self.ls_prob = cp.Problem(self.ls_objective, self.ls_constraints)
        self.max_nullity = min(self.num_total_dofs, self.m * self.num_arms)
        self.resid_null_param = cp.Parameter((self.num_total_dofs, self.max_nullity),name="nullspace")
        self.resid_q_dot_part_param = cp.Parameter(self.num_total_dofs,name="qdot_part")
        self.resid_var = cp.Variable(self.max_nullity,name="residual")

    def get_q_dot(self, target: np.ndarray):
        pass

    def get_ls_soln(self, left_link_name, right_link_name,
        left_desired_twist, right_desired_twist
    ):
        current_config = self.robot_model.getConfig()
        avail_jac = self.get_jacobian(left_link_name, right_link_name)
        self.ls_jac_param.value = avail_jac
        self.ls_config_param.value = current_config
        b_theta = current_config[self.base_dofs[-1]]
        self.c_b_theta_param.value = np.cos(b_theta)
        self.s_b_theta_param.value = np.sin(b_theta)
        self.ls_target_twist_param.value = np.concatenate((
            left_desired_twist, right_desired_twist ))
        res = self.ls_prob.solve()
        return self.ls_q_dot_var.value, avail_jac, res

    def build_constraints(self, q_dot, cfg, c_b_theta, s_b_theta):
        """Build up the constraints for the CP that computes
        the optimal q_dot for the unified controller.
            cfg: cp.Parameter
                Rmk, since this is a cp.Parameter, it doesn't
                actually contain values, so numpy operations
                such as cos cannot be performed on it. Thus,
                cos and sin are given separately
            c_b_theta: cp.Parameter
                cos of the yaw of the base
            s_b_theta: cp.Parameter
                cos of the yaw of the base
        """
        # Non-holonomic constraint on base movement
        # q_dot_y cos(theta) - q_dot_x sin(theta) = 0
        con = [
            #TODO make DPP
            q_dot[-2] * c_b_theta - q_dot[-3] * s_b_theta == 0
        ]
        # Velocity limit on the base
        base_vel_lims = trina.settings.base_velocity_limits()
        con.append(cp.norm(q_dot[-3:-1]) <= base_vel_lims[0])
        con.append(cp.abs(q_dot[-1]) <= base_vel_lims[1])
        # Abs velocity limits
        for i, v_lim in enumerate(trina.settings.limb_velocity_limits()):
            con.append(cp.abs(q_dot[i]) <= v_lim)
            con.append(cp.abs(q_dot[i + len(self.left_dofs)]) <= v_lim)
        #TODO make DPP
        con.extend(self.position_constraints(q_dot, cfg))
        # Avoid most imminent collision
        # s = time.monotonic()
        # jac_a, jac_b, dist, vec = self.get_closest_jac()
        # print(f"Col time: {time.monotonic() - s}")
        # vec = np.array(vec)
        # if np.linalg.norm(vec) > 0:
        #   vec /= np.linalg.norm(vec)
        # vel_a = jac_a @ q_dot
        # vel_b = jac_b @ q_dot
        # # relative speed of points towards each other
        # col_speed = vel_a @ vec - vel_b @ vec
        # accel_lim = 2.0
        # con.append(col_speed <= (2 * accel_lim * dist)**0.5)
        return con

    def position_constraints(self, q_dot, cfg):
        con = []
        ang_accel_lim = 0.5
        upper_lims = trina.settings.limb_position_upper_limits()
        # TODO make DPP
        for i, u_lim in enumerate(upper_lims):
            left_u_lim = u_lim
            if i == 2: # Constrain elbow to bend out
                left_u_lim = 0.1
            left_u_vel_lim = (2 * ang_accel_lim
                    * cp.max(left_u_lim - cfg[self.left_dofs[i]], 0))**0.5
            con.append(q_dot[i] <= left_u_vel_lim)
            con.append(q_dot[i + len(self.left_dofs)]
                <= (2 * ang_accel_lim
                    * (u_lim - cfg[self.right_dofs[i]]))**0.5 )
        for i, l_lim in enumerate(trina.settings.limb_position_lower_limits()):
            con.append(q_dot[i]
                >= -(2 * ang_accel_lim
                    * (cfg[self.left_dofs[i]] - l_lim))**0.5 )
            right_l_lim = l_lim
            if i == 2: # Constrain elbow to bend out
                right_l_lim = -0.1
            right_l_vel_lim = (2 * ang_accel_lim
                    * cp.max(cfg[self.right_dofs[i]] - l_lim, 0))**0.5
            con.append(q_dot[i + len(self.left_dofs)]
                >= -right_l_vel_lim)
        return con

    def get_jacobian(self, left_link_name, right_link_name):
        # Jacobian (column) order is left arm, right arm, base
        left_full_robot_jac = np.vstack((
            self.robot_model.link(left_link_name).getOrientationJacobian(),
            self.robot_model.link(left_link_name).getPositionJacobian([0,0,0])))
        right_full_robot_jac = np.vstack((
            self.robot_model.link(right_link_name).getOrientationJacobian(),
            self.robot_model.link(right_link_name).getPositionJacobian([0,0,0])))
        left_jac = self.pack_jac(left_full_robot_jac)
        right_jac = self.pack_jac(right_full_robot_jac)
        return np.vstack([ left_jac, right_jac ])

    def pack_jac(self, klampt_jac: np.ndarray) -> np.ndarray:
        """Given the klampt format jacobian for a point, pack it so that
        the columns correspond to this module's convention
        (left arm, right arm, base).

        Args:
            klampt_jac (np.ndarray): Klampt format jacobian

        Returns:
            np.ndarray: Packed jacobian fitting module's convention.
        """
        total_dofs = len(self.left_dofs) + len(self.right_dofs) + len(self.base_dofs)
        klampt_jac = np.array(klampt_jac)
        avail_jac = np.zeros((len(klampt_jac), total_dofs))
        for i, ind in enumerate(self.left_dofs):
            avail_jac[:, i] = klampt_jac[:, ind]
        for i, ind in enumerate(self.right_dofs):
            avail_jac[:, i + len(self.left_dofs)] = klampt_jac[:, ind]
        for i, ind in enumerate(self.base_dofs):
            avail_jac[:, i + len(self.left_dofs) + len(self.right_dofs)] = \
                klampt_jac[:, ind]
        return avail_jac

    def get_null_basis(self, s: np.ndarray, vh: np.ndarray):
        v: np.ndarray = vh.T
        v_extra_cols = max(v.shape[1] - s.shape[0], 0)
        nullity = v_extra_cols
        null_inds = []
        for i, val in enumerate(s):
            if val == 0:
                nullity += 1
                null_inds.append(i)
        null_basis = np.empty((v.shape[0], nullity))
        null_basis[:, :v_extra_cols] = v[:, -v_extra_cols:]
        for i, ind in null_inds:
            null_basis[:, v_extra_cols + i] = v[:, ind]
        return null_basis
