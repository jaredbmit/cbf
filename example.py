import numpy as np
import sympy
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# TODO The base trajectory follower control policy is poor when far away from the reference trajectory
    # Which is inevitable for avoiding a big object with CBF
    # Also, phase lag
# TODO use control lyapunov function instead of PD controller


class ControlPolicy:
    def __init__(self):
        return


class ReferenceTrajectoryFollower(ControlPolicy):
    """
    This control policy follows a reference trajectory
    At a given point in time, calculates the first order approximation of 
        control effort required to arrive at the next reference trajectory location
    """
    def __init__(self):
        super().__init__()
        self.time = None
        self.trajectory = None
        self.loaded = False
    
    def load_trajectory(self, time : np.ndarray, trajectory : np.ndarray):
        """
        Necessary initialization for control policy
        Args:
            - time (np.ndarray): (n,) time series
            - trajectory (np.ndarray): (n,d) trajectory
        """
        if not time.ndim == 1 \
            or not trajectory.ndim == 2 \
            or not len(time) == len(trajectory):
            raise ValueError('Requires time is shape (n,) and trajectory is shape (n,d)')

        self.time = time
        self.trajectory = trajectory
        self.loaded = True

    def control(self, time : float, state : np.ndarray):
        """
        TODO This is hardcoded for x,y double integrator dynamics
        Adjust this to incorporate arbitrary dynamic systems etc

        Args:
            - time (float): timestep
            - state (np.ndarray): (n,) current state
        Returns: 
            - u (np.ndarray): (m,) control effort
        """
        if not self.loaded:
            raise AssertionError(f'{type(self).__name__} requires a trajectory to be loaded first')

        # Find the next step in the trajectory
        idx = np.min(np.argwhere(self.time > time))
        p_des = self.trajectory[idx]

        # PD Controller gains
        Kp = np.array([1., 1.])  # Proportional gains for position
        Kd = np.array([1., 1.])  # Derivative gains for velocity

        # Compute the error in position and velocity
        dt = self.time[idx] - time
        p_err = p_des - state[:2]
        v_des = (p_des - self.trajectory[idx - 1]) / dt
        v_err = v_des - state[2:]

        # Control input (acceleration) using PD control
        u = Kp * p_err + Kd * v_err
        print(p_err)
        print(v_err)
        print(u)

        return u


# Create abstract classes for dynamics, barrier functions, alpha functions, control policies
# Can provide access to control barrier function constraint online

# class Field:
#     def __init__(self):
#         raise NotImplementedError

# class ScalarField(Field):
#     def __init__(self):
#         raise NotImplementedError
    
# class Function(ScalarField):
#     def __init__(self):
#         raise NotImplementedError


# -- Main -- #

def example(
    time,
    trajectory,
):
    """
    # Goal: 
    # Given an input trajectory
    # Given a cbf constraint
    # Solves a QP over a time horizon where a reference u is determined to follow the input trajectory
    """
    n, _ = trajectory.shape

    # Initialize dynamical system
    # dx_dt = f(x) + g(x)u
    state = sympy.Matrix(sympy.symbols('x y vx vy'))
    f = state[2:,:].col_join(sympy.zeros(2,1))
    g = sympy.zeros(2).col_join(sympy.eye(2))

    # Make it callable
    f_eval = sympy.utilities.lambdify(state, f, 'numpy')
    g_eval = sympy.utilities.lambdify(state, g, 'numpy')
    def dx_dt(x, u):
        return (f_eval(*x) + g_eval(*x).dot(u.reshape(-1,1))).flatten()

    # Initialize control policy
    trajectory_follower = ReferenceTrajectoryFollower()
    trajectory_follower.load_trajectory(time, trajectory)

    # Define barrier function
    pos_obs = sympy.Matrix([0, 0])
    rad_obs = 0.25
    pos_obs_to_state = state[:2,:] - pos_obs
    b = pos_obs_to_state.dot(pos_obs_to_state) - rad_obs**2

    # Calculate CBF expressions
    # Using identity alpha functions
    input = sympy.Matrix(sympy.symbols('ux uy'))
    cbf = sympy.diff(b, state).dot(f) + sympy.diff(b, state).dot(g * input) + 1 * b
    cbf_2 = sympy.diff(cbf, state).dot(f) + sympy.diff(cbf, state).dot(g * input) + 1 * cbf

    # Initialize optimization problem
    input_reference = sympy.Matrix(sympy.symbols('ux_ref uy_ref'))
    W = sympy.eye(2)
    cost = sympy.sqrt((input - input_reference).dot(W * (input - input_reference)))
    
    # Follow trajectory, with control barrier function
    x = np.concatenate([trajectory[0], np.zeros((2,))])
    x_history = np.empty((n, 4))
    x_history[0] = x.copy()
    for i in range(n - 1):

        # Calculate reference control effort
        u_ref = trajectory_follower.control(time[i], x)

        # Form objective and constraint
        cost_formed = cost.subs([(input_reference[j], u_ref[j]) for j in range(len(u_ref))])
        cbf_formed = cbf_2.subs([(state[j], x[j]) for j in range(len(x))])

        # Lambdify
        cost_eval = sympy.utilities.lambdify(input, cost_formed, 'numpy')
        cbf_eval = sympy.utilities.lambdify(input, cbf_formed, 'numpy')
        def cbf_func(vars):
            return cbf_eval(*vars)
        def cost_func(vars):
            return cost_eval(*vars)
        
        # Solve
        result = scipy.optimize.minimize(
            cost_func, 
            u_ref, # Initial guess
            constraints=[{'type': 'ineq', 'fun': cbf_func}],
        )
        u = result.x
        print(result.success, result.message, result.x)

        # Integrate dynamics
        x += dx_dt(x, u) * (time[i+1] - time[i])

        # Bookkeep
        x_history[i+1] = x

    # Visualize
    cmap = plt.cm.viridis
    norm = Normalize(vmin=time[0], vmax=time[-1])
    points = x_history[:,:2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(time)
    lc.set_linewidth(2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    cb = plt.colorbar(lc, ax=ax)
    cb.set_label('Time')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(x_history[:,0].min(), x_history[:,0].max())
    ax.set_ylim(x_history[:,1].min(), x_history[:,1].max())
    plt.show()


# -- Executable -- #

if __name__ == '__main__':
    # Script inputs
    n = 100
    time = np.arange(n)
    trajectory = np.linspace(-1*np.ones(2), np.ones(2), num=n)

    # Main
    example(
        time,
        trajectory,
    )