import numpy as np
import casadi as ca
class nmpc:
    """
    Todo: seperate nmpc with objective funtion
    nonlinear mpc
    """
    def __init__(self, sys, mode = 'TT', N = None):
        self.sys = sys
        self.opti = ca.Opti()
        self.N = N if N is not None else self.sys.optParam['N']
        self.h = self.sys.optParam['h']
        self.Q = np.array(self.sys.optParam['Q'])
        self.R = np.array(self.sys.optParam['R'])
        self.Q_R = np.array(self.sys.optParam['Q_R'])
        self.d = self.sys.sets['d']

        self.f = self.sys.model
        self.opt_controls =  self.opti.variable(self.N, self.sys.ctrl_dim)
        self.opt_states =  self.opti.variable(self.N, self.sys.obs_dim)
        self.robot_state =  self.opti.variable(self.N, 3)
        self.opti.set_initial(self.robot_state, np.zeros((self.N, 3)))

        self.ctrl_guess, self.state_guess = \
        np.zeros((self.N, self.sys.ctrl_dim)), np.zeros((self.N, self.sys.obs_dim))
        
        self.opt_x0 =  self.opti.parameter(self.sys.obs_dim)
        self.opt_xs =  self.opti.parameter(self.sys.obs_dim)
        self.opt_psi0 =  self.opti.parameter(1)
        self.opt_psis =  self.opti.parameter(1)

        self.tgt_traj =  self.opti.parameter(self.N, self.sys.obs_dim)

        self.obj, self.cost = self.getTTCost()   
        self.opti.minimize(self.obj)
        self.opti.subject_to(self.getConstraints())
        self.opti.subject_to(self.getStateCtrlBounds())
        self.opti.subject_to(self.getTWConstraints())

        opts_setting = {'ipopt.max_iter':1000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6} 
        self.opti.solver('ipopt', opts_setting)

    def setLinearInitialValues(self, init_state, final_state):
        """
        Generate linear initial value
        @param init_state: array, inital state
        @param final_state: array, target state
        @return: array, state inital guess
        """
        state_init_guess = np.zeros((self.N, self.sys.obs_dim))
        for i in range(self.N):
            state_init_guess[i,:] = init_state + (i/( self.N - 1))* (final_state-init_state)
        return state_init_guess

    def getTTCost(self):
        """
        Set up objective function for standard MPC formulation
        return: objective expression
        """
        obj, cost = 0, 0
        for i in range(self.N-1):
            obj +=  (ca.mtimes([(self.opt_states[i, :]-self.opt_xs.T), self.Q, (self.opt_states[i, :]-self.opt_xs.T).T])
                    + ca.mtimes([self.opt_controls[i, :], self.R, self.opt_controls[i, :].T]))
            obj +=  (ca.mtimes([(self.robot_state[i+1, :2]-self.robot_state[i, :2]), self.Q_R[:2,:2], (self.robot_state[i+1, :2]-self.robot_state[i, :2]).T]))
            # try to make the cost more resonable in terms of the distance traveled by the robot
            # obj +=  (ca.mtimes([(self.robot_state[i+1, :]-self.robot_state[-1, :]), self.Q_R * 1.0 / (i/2.0 + 1), (self.robot_state[i+1, :]-self.robot_state[-1, :]).T]))
        obj +=  (ca.mtimes([(self.opt_states[-1, :]-self.opt_xs.T), self.Q, (self.opt_states[-1, :]-self.opt_xs.T).T])
                    + ca.mtimes([self.opt_controls[-1, :], self.R, self.opt_controls[-1, :].T]))

        a = 0.6
        for i in range(self.N-1):
            cost += a * ca.sqrt(ca.mtimes([(self.opt_states[i+1, :2]-self.opt_states[i, :2]), self.Q[:2,:2], (self.opt_states[i+1, :2]-self.opt_states[i, :2]).T]))
            cost += (1-a) * ca.sqrt(ca.mtimes([(self.robot_state[i+1, :2]-self.robot_state[i, :2]), self.Q_R[:2,:2], (self.robot_state[i+1, :2]-self.robot_state[i, :2]).T]))
            # cost = ca.mtimes([self.opt_controls[i, :], self.R*100, self.opt_controls[i, :].T])
            
            # cost += 0.05*ca.fabs(self.opt_states[i, 2]-self.robot_state[i, 2])
            # cost +=  ca.mtimes([self.opt_controls[i, :], self.R, self.opt_controls[i, :].T])
        # obj += (ca.mtimes([(self.opt_states[self.N-1, :]-self.opt_xs.T), self.Q_N, (self.opt_states[self.N-1, :]-self.opt_xs.T).T]))
        return obj, cost

    def getConstraints(self):
        """
        Define collocation and boundary constraints
        return: constraint expression
        """
        ceq = []
        for i in range(self.N - 1): 
            ceq.append(self.getCollocationConstraints(self.opt_states[i,:],self.opt_states[i+1,:],
                                                    self.f(self.opt_states[i, :], self.opt_controls[i, :]),
                                                    self.h))
        ceq.extend(self.getBoundaryConstrainsts(self.opt_states[0,:],self.opt_states[-1,:]))
        return ceq

    def getTWConstraints(self):
        ceq = []
        ceq.append(self.robot_state[-1, 2] == self.opt_psis)
        ceq.append(self.robot_state[0, 2] == self.opt_psi0)
        ceq.append(self.robot_state[-1, 0] == self.opt_states[-1, 0] - self.d*np.cos(self.robot_state[-1, 2]))
        ceq.append(self.robot_state[-1, 1] == self.opt_states[-1, 1] - self.d*np.sin(self.robot_state[-1, 2]))
        ceq.append(self.robot_state[0, 0] == self.opt_states[0, 0] - self.d*np.cos(self.robot_state[0, 2]))
        ceq.append(self.robot_state[0, 1] == self.opt_states[0, 1] - self.d*np.sin(self.robot_state[0, 2]))
        ceq.append(self.opti.bounded(-1, self.opt_states[:, 2] - self.robot_state[:, 2], 1))
        slack_d2 = (self.d * 1.2 )**2
        ceq.append(self.opti.bounded(-slack_d2, (self.opt_states[:, 0] - self.robot_state[:, 0])**2 + (self.opt_states[:, 1] - self.robot_state[:, 1])**2, slack_d2))
        return ceq

    def getCollocationConstraints(self,state1,state2,model,h):
        """
        Define collocation constraint
        @state1: array, current state
        @state2: array, next state
        @model: array, dynamics
        @h: double, step time
        return: collocation constraint expression
        """
        return (state2 == state1 + h * model)

    def getBoundaryConstrainsts(self,x0,xf):
        """
        Define Boundary constraints for x0 and xf
        @x0: array, initial state
        @xf: array, end state
        return: boundary state constraint expression
        """
        ceq = []
        for i in range(self.sys.obs_dim): 
            ceq.extend([(x0[i] == self.opt_x0[i])]) 
            if i in self.sys.sets['idx']:
                # print("terminal constraint on ", i)
                ceq.extend([(xf[i] == self.opt_xs[i])]) # constraint final state to 0
        return ceq

    def getStateCtrlBounds(self):
        """
        Set constraints on state and control
        return: constraints expression
        """
        c = [] 
        for i in range(self.sys.obs_dim):
            c.extend([self.opti.bounded(self.sys.sets['state_min'][i], self.opt_states[:, i], self.sys.sets['state_max'][i])])

        for i in range(self.sys.ctrl_dim):
            c.extend([self.opti.bounded(self.sys.sets['u_min'][i], self.opt_controls[:, i], self.sys.sets['u_max'][i])])
        return c

    def interpolate_state_ctrl(self, x, u, dt):
        """
        Interpolate state and control 
        @x: array, state
        @u: array, control
        @dt: float, control time step
        return: (ctrl, state), u and x after interpolation
        """
        N_intp = int(self.h/dt)
        if N_intp == 1:
            return u, x
        t_full = np.linspace(0, self.h*self.N-dt, self.N*N_intp)
        N_full = len(t_full)
        ctrl_full = np.zeros((N_full, self.sys.ctrl_dim))
        states_full = np.zeros((N_full, self.sys.obs_dim))
        
        for k in range(self.N-1):
            # print("current k {}, N_intp {}".format(k,N_intp ))
            f_k = self.f(x[k], u[k])
            f_k1 = self.f(x[k+1], u[k+1])
            for i in range(N_intp):
                # print("current i {}".format(i))
                idx = k*N_intp+i
                tau = i*dt
                ctrl_full[idx] = u[k] + tau /self.h * (u[k+1] - u[k])
                states_full[idx] =  x[k] + f_k*tau + tau**2/(2*self.h)*(f_k1-f_k)
        # print("states_full: ", states_full) 
        return ctrl_full, states_full

    def shift_sol(self, u_sol, x_sol, steps = 1):
        """
        Shift solution from optimizer as new initial guess
        @u_sol: array, control output from optimizer
        @x_sol: array, state output from optimizer
        @steps: int, shifting steps
        """
        u_tile = [u_sol[-1]]*steps
        x_tile = [x_sol[-1]]*steps
        u_shift = np.concatenate((u_sol[steps:], u_tile))
        x_shift = np.concatenate((x_sol[steps:], x_tile))
        return u_shift, x_shift

    def step(self, stateCurrent, stateTarget = np.zeros(4), UGuess = [], stateGuess = [], initMethod = 'linear', interpolation = False, ctrl_dt=0.02):
        """
        Optimization step for fix point stablization
        """
        # print(f"stateTarget: {stateTarget }")

        self.opti.set_value(self.opt_x0, stateCurrent[0:3])
        self.opti.set_value(self.opt_xs, stateTarget[0:3])
        self.opti.set_value(self.opt_psi0, stateCurrent[3])
        self.opti.set_value(self.opt_psis, stateTarget[3])

        # print("target state: ", self.opti.value(self.opt_xs))
        if stateGuess == []:
            if initMethod == 'linear':
                state_init_guess = self.setLinearInitialValues(stateCurrent[0:3], stateTarget[0:3])
                ctrl_init_guess = self.ctrl_guess
            elif initMethod == 'shift':
                # print("use shifted initial guess")
                ctrl_init_guess, state_init_guess = self.ctrl_guess, self.state_guess
        # print("state_init_guess: ", state_init_guess)
        if UGuess != []:
            ctrl_init_guess = UGuess
        if stateGuess != []:
            state_init_guess = stateGuess

        self.opti.set_initial(self.opt_controls, ctrl_init_guess)
        self.opti.set_initial(self.opt_states, state_init_guess)
        return self.solve(interpolation, dt = ctrl_dt)

    def solve(self,interpolation,dt):
        """
        Solve the optimization problem
        """
        try:      
            sol =  self.opti.solve()
            self.ctrl_sol = sol.value(self.opt_controls)
            self.state_sol = sol.value(self.opt_states)
            self.obj_value = sol.value(self.obj)
            self.cost_value = sol.value(self.cost)
            # print(f"obj_value: {self.obj_value }, cost_value: {self.cost_value}")
            self.robot_state_sol = sol.value(self.robot_state)
            self.ctrl_guess, self.state_guess = self.shift_sol(self.ctrl_sol, self.state_sol)
            if interpolation:  
                self.ctrl_full, self.states_full = self.interpolate_state_ctrl(self.state_sol, self.ctrl_sol, dt = dt)
            else:
                self.ctrl_full, self.states_full = self.ctrl_sol, self.state_sol
        except:
            print("Infeasible Problem Detected!")
            return False
        return True


 
    