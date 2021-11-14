import json
import math
import time
from typing import Dict, List, Tuple
import klampt
from klampt.math import se3, so3
from klampt.model import ik, collide
import cvxpy as cp
import numpy as np
from consts import SETTINGS_PATH


class Tracker:
    def __init__(self, world_model, dt: float, lam: Dict[str, float]=None):
        """Create an optimization problem that generates joint motions to
        achieve a desired hand twist.

        Args:
            world_model (klampt.WorldModel): world model of klampt
        """
        with open(SETTINGS_PATH, "r") as f:
            self.settings = json.load(f)
        self.dt = dt
        self.left_name = "left_tool_link"
        self.right_name = "right_tool_link"
        self.base_name = "base_link"
        self.left_handle_name = "lefthandle_link"
        self.right_handle_name = "righthandle_link"
        self.w_base_name = "base_link"
        self.world_model: klampt.WorldModel = world_model
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
        self.num_arm_dofs = len(self.left_dofs) + len(self.right_dofs)
        self.num_total_dofs = self.num_arm_dofs + len(self.base_dofs)
        self._init_config()
        self.t_lr = se3.mul(se3.inv(
            self.robot_model.link(self.left_name).getTransform()),
            self.robot_model.link(self.right_name).getTransform())
        self.wheelchair_dofs = self.settings["wheelchair_dofs"]

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
        q = self.robot_model.getConfig()
        self.num_klampt_dofs = len(q)
        self.left_attractor = np.array(self.get_arm_cfg('l', q))
        self.right_attractor = np.array(self.get_arm_cfg('r', q))
        self.arms_attractor = np.concatenate((self.left_attractor, self.right_attractor))
        self.left_in_right = se3.mul(
            se3.inv(self.robot_model.link("left_tool_link").getTransform()),
            self.robot_model.link("right_tool_link").getTransform())
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
        self.arms_config_param = cp.Parameter(self.num_arm_dofs,
            name="arms_config")
        self.q_dot_part_param = cp.Parameter(self.num_total_dofs,
            name="qdot_part")
        self.resid_var = cp.Variable(self.max_nullity, name="residual")
        self.q_dot_full = (self.q_dot_part_param
            + self.null_basis_param @ self.resid_var)
        self.resid_constraints = self.build_constraints(self.q_dot_full)
        ### Null space objective weighting
        if lam is None:
            lam = {}
        self.arm_penalty = lam.get("arm_penalty", 0)
        self.strafe_penalty = lam.get("strafe_penalty", 0)
        self.base_penalty = lam.get("base_penalty", 0)
        self.attractor_penalty = lam.get("attractor_penalty", 0)
        self.resid_obj = cp.Minimize(
            # Arm dofs
            self.arm_penalty * cp.norm2(self.q_dot_full[:self.num_arm_dofs])**2
            # Strafe dof
            + self.strafe_penalty * self.q_dot_full[self.num_arm_dofs + 1]**2
            # Base dofs
            + self.base_penalty * cp.norm2(self.q_dot_full[self.num_arm_dofs:])**2
            # Attractor configs
            + self.attractor_penalty * cp.norm2(self.q_dot_full[:self.num_arm_dofs] * self.dt + self.arms_config_param - self.arms_attractor)**2
        )
        self.resid_prob = cp.Problem(self.resid_obj, self.resid_constraints)
        self.collider = collide.WorldCollider(self.world_model)
        self._ignore_collision_pairs()

    def get_target_config(self, vel: np.ndarray) -> List[float]:
        orig_w_cfg = self.wheelchair_model.getConfig()
        w_t_wb = self.wheelchair_model.link(self.w_base_name).getTransform()
        yaw = math.atan2(w_t_wb[0][1], w_t_wb[0][0])
        delta_yaw = vel[1] * self.dt
        delta_p = (np.array([np.cos(yaw) * vel[0], np.sin(yaw) * vel[0], 0])
            * self.dt)
        w_q = orig_w_cfg[:]
        w_q[self.wheelchair_dofs[0]] += delta_p[0]
        w_q[self.wheelchair_dofs[1]] += delta_p[1]
        w_q[self.wheelchair_dofs[2]] += delta_yaw
        self.wheelchair_model.setConfig(w_q)
        collides = False
        for _ in self.collider.collisions():
            collides = True
            break
        if collides:
            return "collision"
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

    def get_configs(self) -> Tuple[List[float], List[float]]:
        return self.robot_model.getConfig(), self.wheelchair_model.getConfig()

    def set_configs(self, cfgs: Tuple[List[float], List[float]]):
        self.robot_model.setConfig(cfgs[0])
        self.wheelchair_model.setConfig(cfgs[1])

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
        q_dot_part, _ = self.get_ls_soln(cfg, target)
        q_dot_h, _ = self.get_resid_soln(cfg, q_dot_part)
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
        j_cfg = self.get_arms_cfg(cfg)
        self.p_upper_lim_param.value = np.sqrt(
            np.maximum(2 * self.a_lim * (self.full_q_upper_lim - j_cfg), 0)
        )
        self.p_lower_lim_param.value = -np.sqrt(
            np.maximum(2 * self.a_lim * (j_cfg - self.full_q_lower_lim), 0)
        )

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
        np_cfg = np.array(cfg)
        self.arms_config_param.value = np.concatenate((
            np_cfg[self.left_dofs], np_cfg[self.right_dofs]))

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
        left_full_robot_jac = np.concatenate((
            np.array(self.robot_model.link(self.left_name).getOrientationJacobian()),
            np.array(self.robot_model.link(self.left_name).getPositionJacobian([0,0,0]))
        ))
        right_full_robot_jac = np.concatenate((
            np.array(self.robot_model.link(self.right_name).getOrientationJacobian()),
            np.array(self.robot_model.link(self.right_name).getPositionJacobian([0,0,0]))
        ))
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

    def pack_cfg(self, cfg: List[float], n_cfg: np.ndarray) -> List[float]:
        """Pack a new configuration for the DoFs the tracker can modify
        (left arm, right arm, base).

        Args:
            cfg (List[float]): Klampt format config template.
            n_cfg (np.ndarray): New values of tracker dofs.

        Returns:
            List[float]: Klampt format config.
        """
        for i, d in enumerate(self.left_dofs):
            cfg[d] = n_cfg[i]
        for i, d in enumerate(self.right_dofs):
            cfg[d] = n_cfg[i + self.m]
        for i, d in enumerate(self.base_dofs):
            cfg[d] = n_cfg[i + self.num_arms * self.m]

    def extract_cfg(self, cfg: List[float]) -> np.ndarray:
        np_cfg = np.array(cfg)
        return np_cfg[self.left_dofs + self.right_dofs + self.base_dofs]

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

    def _get_t_hee(self) -> tuple:
        # Hand tilted 45 degrees down toward the wheelchair handle
        aa = ((0, 1, 0), np.pi / 4)
        r_hee = so3.from_axis_angle(aa)
        return (r_hee, (-0.05, 0, 0.15))

    def _ignore_collision_pairs(self):
        # Ignore collisions between hands and handles
        gripper_dofs = self.left_gripper_dofs + self.right_gripper_dofs
        for name in [self.left_handle_name, self.right_handle_name]:
            for d in gripper_dofs:
                self.collider.ignoreCollision((
                    self.wheelchair_model.link(name), self.robot_model.link(d)
                ))
        for i in range(self.robot_model.numLinks()):
            for j in range(self.robot_model.numLinks()):
                if i != j:
                    link_a = self.robot_model.link(i)
                    link_b = self.robot_model.link(j)
                    if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                        self.collider.ignoreCollision((link_a, link_b))
        for i in range(self.wheelchair_model.numLinks()):
            for j in range(self.wheelchair_model.numLinks()):
                if i != j:
                    link_a = self.wheelchair_model.link(i)
                    link_b = self.wheelchair_model.link(j)
                    if (not link_a.geometry().empty()) and (not link_b.geometry().empty()):
                        self.collider.ignoreCollision((link_a, link_b))


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


def profile_opt():
    world = klampt.WorldModel()
    world.loadFile("Model/worlds/TRINA_world_cholera.xml")
    robot_model: klampt.RobotModel = world.robot(0)
    with open(SETTINGS_PATH, "r") as f:
        settings = json.load(f)
    # q = robot_model.getConfig()
    # cfg_l = [0.17317459, -1.66203799, -2.25021315, 3.95050542, -0.59267456, -0.8280866]#[-4.635901893690355, 5.806223419961121, 1.082262209315542, -2.753160794116381, 1.0042225011740618, 4.022876408436895]
    # for i, d in enumerate(settings["left_arm_dofs"]):
    #     q[d] = cfg_l[i]
    # cfg_r = mirror_arm_config(cfg_l)
    # for i, d in enumerate(settings["right_arm_dofs"]):
    #     q[d] = cfg_r[i]
    # robot_model.setConfig(q)
    weights = {
        "arm_penalty": 2,
        "strafe_penalty": 2,
        "base_penalty": 1,
        "attractor_penalty": 10000
    }
    dt = 1 / 50
    t = Tracker(world, dt, weights, True)
    print("Start left tf: ", t.robot_model.link(t.left_name).getTransform())
    print("Start right tf: ", t.robot_model.link(t.right_name).getTransform())
    q = t.robot_model.getConfig()
    target_twist = np.array([0,0,0,1,0,0,0,0,0,1,0,0])
    q_dot = t.get_q_dot(q, target_twist)
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
    for i in range(runs):
        q_dot = t.get_q_dot(q, target_twist)
        red_q += q_dot * dt
        for i, d in enumerate(settings["left_arm_dofs"]):
            q[d] = red_q[i]
        for i, d in enumerate(settings["right_arm_dofs"]):
            q[d] = red_q[i + 6]
        for i, d in enumerate(settings["base_dofs"]):
            q[d] = red_q[i + 12]
    delta = time.time() - start
    print(f"{runs} runs took {delta}s, for {delta/runs}s on average")
    print(f"After {runs} steps, config moved from \n{start_red_q} \nto \n{red_q}, \nfor a delta of \n{red_q - start_red_q}")
    print(f"Delta has a infinity norm of {np.linalg.norm(red_q - start_red_q, np.inf)}")
    print("End left tf: ", t.robot_model.link(t.left_name).getTransform())
    print("End right tf: ", t.robot_model.link(t.right_name).getTransform())


def test_wheelchair_update():
    from klampt import vis
    world = klampt.WorldModel()
    world.loadFile("Model/worlds/TRINA_world_cholera.xml")
    with open(SETTINGS_PATH, "r") as f:
        settings = json.load(f)
    vis.add("world", world)
    dt = 1 / 50
    weights = {
        "arm_penalty": 0,
        "strafe_penalty": 1,
        "base_penalty": 0,
        "attractor_penalty": 10
    }
    t = Tracker(world, dt, lam=weights)
    timesteps = 1000
    vis.show()
    for _ in range(timesteps):
        cfgs = t.get_configs()
        res = t.get_target_config(np.array([1.0, 0.0]))
        if res != "success":
            print(res)
            t.set_configs(cfgs)
        time.sleep(dt)
    # for _ in range(timesteps):
    #     print(t.get_target_config(np.array([1.0, -0.5])))
    #     time.sleep(dt)


if __name__ == "__main__":
    # profile_opt()
    test_wheelchair_update()
