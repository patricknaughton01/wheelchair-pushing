import json
import math
import time
from typing import Dict, List
import klampt
import klampt.plan as kp
from klampt.math import vectorops as vo
import cvxpy as cp
import numpy as np
from numpy.core.numeric import full
from consts import SETTINGS_PATH


class Tracker:
    def __init__(self, world_model, lam: Dict[str, float], lock_arms: bool=True):
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
        self.world_model: klampt.WorldModel = world_model
        self.robot_model: klampt.RobotModel = self.world_model.robot(0)
        self.cspace = kp.robotcspace.RobotCSpace(self.robot_model)
        self.left_dofs: List[int] = self.settings["left_arm_dofs"]
        self.right_dofs: List[int] = self.settings["right_arm_dofs"]
        self.base_dofs: List[int] = self.settings["base_dofs"]
        self.num_arm_dofs = len(self.left_dofs) + len(self.right_dofs)
        self.num_total_dofs = self.num_arm_dofs + len(self.base_dofs)

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
        self.a_lim = 1

        # Unified controller optimization problems
        self.m = 6  # Dimensionality of a twist
        self.num_arms = 2
        self.num_klampt_dofs = len(self.robot_model.getConfig())
        ## Initial LS problem
        self.ls_target_twist_param = cp.Parameter(
            self.m * self.num_arms, name="v_target")
        self.jac_param = cp.Parameter(
            (self.m * self.num_arms, self.num_total_dofs), name="jacobian")
        self.ls_q_dot_var = cp.Variable(self.num_total_dofs, name="qdot")
        self.ls_objective = cp.Minimize(
            cp.norm2(
                self.jac_param @ self.ls_q_dot_var
                    - self.ls_target_twist_param) ** 2
        )
        self.p_upper_lim_param = cp.Parameter(
            len(self.left_dofs) + len(self.right_dofs),
            name="p_upper_lim_param")
        self.p_lower_lim_param = cp.Parameter(
            len(self.left_dofs) + len(self.right_dofs),
            name="p_lower_lim_param")
        self.constraints = self.build_constraints(self.ls_q_dot_var)
        self.ls_prob = cp.Problem(self.ls_objective, self.constraints)
        ## Null space optimization
        self.max_nullity = min(self.num_total_dofs, self.m * self.num_arms)
        self.null_basis_param = cp.Parameter(
            (self.num_total_dofs, self.max_nullity), name="nullspace")
        self.q_dot_part_param = cp.Parameter(self.num_total_dofs,
            name="qdot_part")
        self.resid_var = cp.Variable(self.max_nullity, name="residual")
        self.q_dot_full = (self.q_dot_part_param
            + self.null_basis_param @ self.resid_var)
        self.resid_constraints = self.build_constraints(self.q_dot_full)
        ### Null space objective weighting
        self.arm_penalty = lam.get("arm_penalty", 0)
        self.strafe_penalty = lam.get("strafe_penalty", 0)
        self.base_penalty = lam.get("base_penalty", 0)
        self.resid_obj = cp.Minimize(
            # Arm dofs
            self.arm_penalty * cp.norm2(self.q_dot_full[:self.num_arm_dofs])**2
            # Strafe dof
            + self.strafe_penalty * self.q_dot_full[self.num_arm_dofs + 1]**2
            # Base dofs
            + self.base_penalty * cp.norm2(self.q_dot_full[self.num_arm_dofs:])**2
        )
        self.resid_prob = cp.Problem(self.resid_obj, self.resid_constraints)

    def get_q_dot(self, cfg: List[float], target: np.ndarray):
        q_dot_part, _ = self.get_ls_soln(cfg, target)
        q_dot_h, _ = self.get_resid_soln(cfg, q_dot_part)
        # print("Particular soln", q_dot_part)
        return q_dot_part + q_dot_h

    def get_ls_soln(self, cfg: List[float], target: np.ndarray):
        self.fill_ls_params(cfg)
        self.ls_target_twist_param.value = target
        res = self.ls_prob.solve()
        return self.ls_q_dot_var.value, res

    def fill_ls_params(self, cfg):
        self.robot_model.setConfig(cfg)
        # Fill in jacobian param
        self.jac_param.value = self.get_jacobian()
        # Set vel limits to enforce position constraints
        j_cfg = self.get_arm_cfg(cfg)
        self.p_upper_lim_param.value = np.sqrt(
            np.maximum(2 * self.a_lim * (self.full_q_upper_lim - j_cfg), 0)
        )
        self.p_lower_lim_param.value = -np.sqrt(
            np.maximum(2 * self.a_lim * (j_cfg - self.full_q_lower_lim), 0)
        )
        # print("Upper lim", np.sqrt(
        #     np.maximum(2 * self.a_lim * (self.full_q_upper_lim - j_cfg), 0)
        # ))
        # print("Lower lim", -np.sqrt(
        #     np.maximum(2 * self.a_lim * (j_cfg - self.full_q_lower_lim), 0)
        # ))

    def get_resid_soln(self, cfg: List[float], part_soln: np.ndarray):
        self.fill_resid_params(cfg, part_soln)
        res = self.resid_prob.solve()
        return self.null_basis_param.value @ self.resid_var.value, res

    def fill_resid_params(self, cfg: List[float], part_soln: np.ndarray):
        self.robot_model.setConfig(cfg)
        jac = self.get_jacobian()
        self.null_basis_param.value = np.zeros(
            (self.num_total_dofs, self.max_nullity))
        null_basis = self.get_null_basis(jac)
        self.null_basis_param.value[:, :null_basis.shape[1]] = null_basis
        self.q_dot_part_param.value = part_soln

    def build_constraints(self, q_dot_var):
        con = []
        # Absolute velocity limits
        con.append(cp.abs(q_dot_var)
            <= np.concatenate((self.v_lim, self.v_lim, self.base_v_lim)))
        # Position limits
        con.append(q_dot_var[:self.num_arm_dofs]
            <= self.p_upper_lim_param)
        con.append(q_dot_var[:self.num_arm_dofs]
            >= self.p_lower_lim_param)
        return con

    def get_jacobian(self) -> np.ndarray:
        """Get the (2 * self.m, self.num_total_dofs) jacobian that has the
        jacobian of the left-hand point as the top self.m rows, then the
        jacobian of the right-hand point.

        Returns:
            np.ndarray: Left jacobian stacked on top of right jacobian
                (in module convention).
        """
        # Jacobian (column) order is left arm, right arm, base
        left_full_robot_jac = np.array(
            self.robot_model.link(self.left_name).getJacobian([0,0,0]))
        right_full_robot_jac = np.array(
            self.robot_model.link(self.right_name).getJacobian([0,0,0]))
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
        klampt_jac = np.array(klampt_jac)
        avail_jac = np.zeros((len(klampt_jac), self.num_total_dofs))
        for i, ind in enumerate(self.left_dofs):
            avail_jac[:, i] = klampt_jac[:, ind]
        for i, ind in enumerate(self.right_dofs):
            avail_jac[:, i + len(self.left_dofs)] = klampt_jac[:, ind]
        for i, ind in enumerate(self.base_dofs):
            avail_jac[:, i + len(self.left_dofs) + len(self.right_dofs)] = \
                klampt_jac[:, ind]
        return avail_jac

    def get_null_basis(self, jac: np.ndarray) -> np.ndarray:
        """For the input jacobian, find a basis for its null space.

        Args:
            jac (np.ndarray): Jacobian (or generic matrix) to get null basis
                for.

        Returns:
            np.ndarray: For jac with shape (M, N), jac = u s vh, v has shape
                (N, N). Return a matrix of shape (N, ?) where ? is the
                dimension of jac's null space.
        """
        _, s, vh = np.linalg.svd(jac)
        v: np.ndarray = vh.T
        return v[:, np.sum(s > 0):]

    def get_arm_cfg(self, cfg: List[float]) -> np.ndarray:
        return np.array(cfg)[self.left_dofs + self.right_dofs]


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


if __name__ == "__main__":
    world = klampt.WorldModel()
    world.loadFile("Model/worlds/TRINA_world_cholera.xml")
    robot_model: klampt.RobotModel = world.robot(0)
    with open(SETTINGS_PATH, "r") as f:
        settings = json.load(f)
    q = robot_model.getConfig()
    cfg_l = [0.17317459, -1.66203799, -2.25021315, 3.95050542, -0.59267456, -0.8280866]#[-4.635901893690355, 5.806223419961121, 1.082262209315542, -2.753160794116381, 1.0042225011740618, 4.022876408436895]
    for i, d in enumerate(settings["left_arm_dofs"]):
        q[d] = cfg_l[i]
    cfg_r = mirror_arm_config(cfg_l)
    for i, d in enumerate(settings["right_arm_dofs"]):
        q[d] = cfg_r[i]
    robot_model.setConfig(q)
    weights = {
        "arm_penalty": 2,
        "strafe_penalty": 10,
        "base_penalty": 1
    }
    t = Tracker(world, weights, True)
    q_dot = t.get_q_dot(q, np.zeros(12))
    print("Final q dot", q_dot)
    full_q_dot = np.zeros(len(q))
    for i, d in enumerate(settings["left_arm_dofs"]):
        full_q_dot[d] = q_dot[i]
    for i, d in enumerate(settings["right_arm_dofs"]):
        full_q_dot[d] = q_dot[i + 6]
    for i, d in enumerate(settings["base_dofs"]):
        full_q_dot[d] = q_dot[i + 12]
    jac = t.get_jacobian()
    print("Final twists: ", jac @ q_dot)
    start_red_q = np.array(q)[settings["left_arm_dofs"] + settings["right_arm_dofs"] + settings["base_dofs"]]
    red_q = np.copy(start_red_q)
    start = time.time()
    runs = 100
    dt = 1 / 50
    for i in range(runs):
        q_dot = t.get_q_dot(q, np.zeros(12))
        red_q += q_dot * dt
    delta = time.time() - start
    print(f"{runs} runs took {delta}s, for {delta/runs}s on average")
    print(f"After {runs} steps, config moved from \n{start_red_q} \nto \n{red_q}, \nfor a delta of \n{red_q - start_red_q}")
