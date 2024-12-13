import numpy as np
import sympy
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


class AffineDynamicalSystem:
    """
    State-space dynamical system formulation in control affine form
    x_dot = f(x) + g(x)u
    """
    def __init__(
        self, 
        x: sympy.matrices.MatrixBase, 
        u: sympy.matrices.MatrixBase, 
        f: sympy.matrices.MatrixBase, 
        g: sympy.matrices.MatrixBase,
    ):
        """
        Args:
            - x (sympy.matrices.MatrixBase): state variables for the dynamical system
            - u (sympy.matrices.MatrixBase): control variables for the dynamical system
            - f (sympy.matrices.MatrixBase): uncontrolled state dynamics function f(x)
            - g (sympy.matrices.MatrixBase): controlled state dynamics function g(x)
        """
        if not isinstance(x, sympy.matrices.MatrixBase):
            raise TypeError(f"x must be a sympy MatrixBase, got {type(x).__name__}")
        if not isinstance(u, sympy.matrices.MatrixBase):
            raise TypeError(f"u must be a sympy MatrixBase, got {type(u).__name__}")
        if not isinstance(f, sympy.matrices.MatrixBase):
            raise TypeError(f"f must be a sympy MatrixBase, got {type(f).__name__}")
        if not isinstance(g, sympy.matrices.MatrixBase):
            raise TypeError(f"g must be a sympy MatrixBase, got {type(g).__name__}")
        
        if x.shape != f.shape or x.shape != (g * u).shape:
            raise ValueError(f"Shape mismatch: x.shape {x.shape} must match f.shape {f.shape} and (g * u).shape {(g * u).shape}")

        self.x = x
        self.u = u
        self.f = f
        self.g = g

        # Lambdify
        dx_dt_expr = f + g * u
        self.dx_dt_eval = sympy.utilities.lambdify((*x, *u), dx_dt_expr, 'numpy')

    def evaluate_dx_dt(
        self, 
        x: np.ndarray, 
        u: np.ndarray,
    ):
        """
        Explicit call to calculate the time derivative of x
        Args:
            - x (np.ndarray): (n,) state values
            - u (np.ndarray): (m,) control values
        Returns:
            - (np.ndarray): (n,) time derivative of x
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy array, not {type(x).__name__}")
        if not isinstance(u, np.ndarray):
            raise TypeError(f"u must be a numpy array, not {type(u).__name__}")
        
        if not self._is_numeric(x):
            raise TypeError(f"x must contain only scalar values")
        if not self._is_numeric(u):
            raise TypeError(f"u must contain only scalar values")
        
        return self.dx_dt_eval(*x, *u)

    def _is_numeric(self, arr):
        return all(
            not isinstance(item, sympy.Basic) and np.isscalar(item) 
            for item in np.ravel(arr)
        )


class ControlBarrierFunction:
    """
    Control barrier function for an affine dynamical system
    """
    def __init__(
        self,
        sys: AffineDynamicalSystem,
        b: sympy.Expr,
        a: sympy.Expr = None,
        max_order: int = 2,
    ):
        """
        Args:
            - sys (AffineDynamicalSystem): dynamical system
            - b (sympy.Expr): barrier function
            - a (sympy.Expr, optional): class-Kappa function
            - max_order(int, optional): maximum order for high-order CBFs
        """
        if a is None:
            a = sympy.Symbol('value')
            
        if not isinstance(sys, AffineDynamicalSystem):
            raise TypeError(f"sys must be an AffineDynamicalSystem, got {type(sys).__name__}")
        if not isinstance(b, sympy.Expr):
            raise TypeError(f"b must be a sympy Expr, got {type(b).__name__}")
        if not isinstance(a, sympy.Expr):
            raise TypeError(f"a must be a sympy Expr, got {type(a).__name__}")
        if not isinstance(max_order, int):
            raise TypeError(f"max_order must be an integer, got {type(max_order).__name__}")
        
        if len(a.free_symbols) != 1:
            raise ValueError(f"a must be an expression of one variable, instead it contains {a.free_symbols}")
        
        self.sys = sys
        self.b = b
        self.a = a

        # Calculate the CBF expression to a potentially high degree
        cbf_expr = self._calculate_cbf_expr(sys, b, a, max_order=max_order)
        print(cbf_expr)
        
        # Lambdify
        self.cbf_eval = sympy.utilities.lambdify((*sys.x, *sys.u), cbf_expr, 'numpy')

    def evaluate_cbf(
        self, 
        x: np.ndarray, 
        u: np.ndarray,
    ):
        """
        Explicit call to calculate the control barrier function
        Args:
            - x (np.ndarray): (n,) state values
            - u (np.ndarray): (m,) control values
        Returns:
            - (float): control barrier function, evaluated
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy array, not {type(x).__name__}")
        if not isinstance(u, np.ndarray):
            raise TypeError(f"u must be a numpy array, not {type(u).__name__}")
        
        if not self._is_numeric(x):
            raise TypeError(f"x must contain only scalar values")
        if not self._is_numeric(u):
            raise TypeError(f"u must contain only scalar values")
        
        return self.cbf_eval(*x, *u)

    def _calculate_cbf_expr(self, sys, b, a, max_order):
        cbf_expr = b
        for i in range(max_order):
            alpha = a.subs([(next(iter(a.free_symbols)), cbf_expr)])
            cbf_expr = sympy.diff(cbf_expr, sys.x).dot(sys.f) + sympy.diff(cbf_expr, sys.x).dot(sys.g * sys.u) + alpha
            if cbf_expr.has(sys.u[0]) and cbf_expr.has(sys.u[1]):
                return cbf_expr
            
        raise ValueError(f"system has a relative degree higher than the maximum order {max_order}")
    
    def _is_numeric(self, arr):
        return all(
            not isinstance(item, sympy.Basic) and np.isscalar(item) 
            for item in np.ravel(arr)
        )


class ControlLyapunovFunction:
    """
    Control Lyapunov function for an affine dynamical system
    """
    def __init__(
        self,
        sys: AffineDynamicalSystem,
        v: sympy.Expr,
        a: sympy.Expr = None,
        relaxation: sympy.Symbol = None,
    ):
        """
        Args:
            - sys (AffineDynamicalSystem): dynamical system
            - b (sympy.Expr): Lyapunov function
            - a (sympy.Expr, optional): class-Kappa function
            - relax (sympy.Symbol, optional): relaxation variable to include in CLF
        """
        if a is None:
            a = sympy.Symbol('value')

        if not isinstance(sys, AffineDynamicalSystem):
            raise TypeError(f"sys must be an AffineDynamicalSystem, got {type(sys).__name__}")
        if not isinstance(v, sympy.Expr):
            raise TypeError(f"v must be a sympy Expr, got {type(v).__name__}")
        if not isinstance(a, sympy.Expr):
            raise TypeError(f"a must be a sympy Expr, got {type(a).__name__}")
        if relaxation is not None and not isinstance(relaxation, sympy.Symbol):
            raise TypeError(f"relaxation must be a sympy Symbol, got {type(relaxation).__name__}")

        if len(a.free_symbols) != 1:
            raise ValueError(f"a must be an expression of one variable, instead it contains {a.free_symbols}")
        
        self.sys = sys
        self.v = v
        self.a = a
        self.relax = False if relaxation is None else True

        # Calculate the CLF expression
        alpha = a.subs([(next(iter(a.free_symbols)), v)])
        clf_expr = sympy.diff(v, sys.x).dot(sys.f) + sympy.diff(v, sys.x).dot(sys.g * sys.u) + alpha
        if self.relax:
            clf_expr += - relaxation
        print(clf_expr)
        
        # Lambdify
        if self.relax:
            self.clf_eval = sympy.utilities.lambdify((*sys.x, *sys.u, relaxation), clf_expr, 'numpy')
        else:
            self.clf_eval = sympy.utilities.lambdify((*sys.x, *sys.u), clf_expr, 'numpy')

    def evaluate_clf(
        self, 
        x: np.ndarray, 
        u: np.ndarray,
        d: float = None,
    ):
        """
        Explicit call to calculate the control Lyapunov function
        Args:
            - x (np.ndarray): (n,) state values
            - u (np.ndarray): (m,) control values
            - d (float, optional): relaxation value
        Returns:
            - (float): control Lyapunov function, evaluated
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy array, not {type(x).__name__}")
        if not isinstance(u, np.ndarray):
            raise TypeError(f"u must be a numpy array, not {type(u).__name__}")
        if not isinstance(d, float):
            raise TypeError(f"d must be a float, not {type(d).__name__}")
        
        if not self._is_numeric(x):
            raise TypeError(f"x must contain only scalar values")
        if not self._is_numeric(u):
            raise TypeError(f"u must contain only scalar values")
        if d is not None and not self.relax:
            raise ValueError(f"d should only be specific with a relaxed CLF, relaxation is set to {self.relax}")
        
        if self.relax:
            return self.clf_eval(*x, *u, d)
        else:
            return self.clf_eval(*x, *u)
    
    def _is_numeric(self, arr):
        return all(
            not isinstance(item, sympy.Basic) and np.isscalar(item) 
            for item in np.ravel(arr)
        )
    

# -- Main -- #

# def example(
#     time,
#     trajectory,
# ):
#     n, _ = trajectory.shape

#     # Define dynamical system
#     state = sympy.Matrix(sympy.symbols('x y vx vy'))
#     input = sympy.Matrix(sympy.symbols('ux uy'))
#     f = state[2:,:].col_join(sympy.zeros(2,1))
#     g = sympy.zeros(2).col_join(sympy.eye(2))

#     # Initialize dynamical system
#     sys = AffineDynamicalSystem(state, input, f, g)

#     # Define barrier function
#     pos_obs = sympy.Matrix([0, 0])
#     rad_obs = 0.3
#     b = (state[:2,:] - pos_obs).dot(state[:2,:] - pos_obs) - rad_obs**2

#     # Initialize CBF
#     a = 1.0 * sympy.Symbol('value')
#     cbf = ControlBarrierFunction(sys, b, a, max_order=2)

#     # Define optimization problem
#     delta = sympy.Symbol('delta')
#     p = 1.
#     W = sympy.eye(2)
#     cost = sympy.sqrt((input).dot(W * (input))) + p * delta ** 2
#     cost_eval = sympy.utilities.lambdify((*input, delta), cost, 'numpy')
#     def cost_func(vars):
#         return cost_eval(*vars)

#     # Bookkeeping
#     x = np.concatenate([trajectory[0], np.zeros((2,))])
#     x_history = np.empty((n, 4))
#     x_history[0] = x.copy()
#     u_history = np.empty((n - 1, 2))

#     for i in range(n - 1):
#         # Current goal position
#         x_goal = trajectory[i + 1]

#         # Define Lyapunov function for local objective
#         K = sympy.diag(1, 0.5)
#         vd = K * (x_goal.reshape((-1,1)) - state[:2,:])
#         V = (state[2:,:] - vd).dot(state[2:,:] - vd)

#         # Initialize CLF
#         a = 1.0 * sympy.Symbol('value')
#         clf = ControlLyapunovFunction(sys, V, a, relaxation=delta)

#         # Form constraint functions
#         cbf_constraint = lambda u: cbf.evaluate_cbf(x, u[0:2]) # Ignore delta
#         clf_constraint = lambda u: - clf.evaluate_clf(x, u[0:2], u[2])

#         # Solve
#         result = scipy.optimize.minimize(
#             cost_func, 
#             [0, 0, 0], # Initial guess
#             constraints=[
#                 {'type': 'ineq', 'fun': cbf_constraint},
#                 {'type': 'ineq', 'fun': clf_constraint},
#             ],
#         ) 
#         u = result.x[:-1]
#         d = result.x[-1]
#         print(result.success, result.message, result.x)

#         # Integrate dynamics
#         x += sys.evaluate_dx_dt(x, u).flatten() * (time[i+1] - time[i])

#         # Bookkeep
#         u_history[i] = u
#         x_history[i+1] = x  

#     # Visualize
#     cmap = plt.cm.viridis
#     norm = Normalize(vmin=time[0], vmax=time[-1])
#     points = x_history[:,:2].reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, cmap=cmap, norm=norm)
#     lc.set_array(time)
#     lc.set_linewidth(2)
#     fig, ax = plt.subplots()
#     ax.add_collection(lc)
#     cb = plt.colorbar(lc, ax=ax)
#     cb.set_label('Time')
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_xlim(x_history[:,0].min(), x_history[:,0].max())
#     ax.set_ylim(x_history[:,1].min(), x_history[:,1].max())
#     plt.show()

# -- Main -- #
# Still phase lag control issues

def example(
    time,
    trajectory,
):
    n, _ = trajectory.shape

    # Initialize dynamical system
    # dx_dt = f(x) + g(x)u
    state = sympy.Matrix(sympy.symbols('x y vx vy'))
    input = sympy.Matrix(sympy.symbols('ux uy'))
    f = state[2:,:].col_join(sympy.zeros(2,1))
    g = sympy.zeros(2).col_join(sympy.eye(2))
    f_eval = sympy.utilities.lambdify(state, f, 'numpy')
    g_eval = sympy.utilities.lambdify(state, g, 'numpy')
    def dx_dt(x, u):
        return (f_eval(*x) + g_eval(*x).dot(u.reshape(-1,1))).flatten()

    # Define Lyapunov function
    state_goal = sympy.Matrix(sympy.symbols('xd yd'))
    K = sympy.diag(1, 1) * 0.2
    V = (state[2:,:] - K * (state_goal - state[:2,:])).dot(state[2:,:] - K * (state_goal - state[:2,:]))
    print(V)

    # Calculate CLF expression, with relaxation
    c1 = 20.
    delta = sympy.symbols('delta')
    clf = - sympy.diff(V, state).dot(f) - sympy.diff(V, state).dot(g * input) - c1 * V + delta

    # Define barrier function
    pos_obs = sympy.Matrix([0.4, 0.6])
    rad_obs = 0.25
    b = (state[:2,:] - pos_obs).dot(state[:2,:] - pos_obs) - rad_obs ** 2

    # Calculate CBF expressions
    c2 = 1.
    cbf = sympy.diff(b, state).dot(f) + sympy.diff(b, state).dot(g * input) + c2 * b
    c3 = 1.
    cbf_2 = sympy.diff(cbf, state).dot(f) + sympy.diff(cbf, state).dot(g * input) + c3 * cbf

    # Initialize optimization problem
    W = sympy.eye(2)
    p = 1
    cost = (input).dot(W * (input)) + p * delta ** 2
    cost_eval = sympy.utilities.lambdify((*input, delta), cost, 'numpy')
    def cost_func(vars):
            return cost_eval(*vars)
    
    # Bookkeeping
    x = np.concatenate([trajectory[0], np.zeros((2,))])
    x_history = np.empty((n, 4))
    x_history[0] = x.copy()
    u_history = np.empty((n - 1, 2))
    
    # Follow trajectory, with control barrier function
    for i in range(n - 1):
        # Current goal position
        x_goal = trajectory[i + 1]

        # Form constraints
        clf_formed = clf.subs(
            [(state[j], x[j]) for j in range(len(x))] \
            + [(state_goal[j], x_goal[j]) for j in range(len(x_goal))]
        )
        cbf_formed = cbf_2.subs([(state[j], x[j]) for j in range(len(x))])

        # Lambdify
        clf_eval = sympy.utilities.lambdify((*input, delta), clf_formed, 'numpy')
        cbf_eval = sympy.utilities.lambdify((*input, delta), cbf_formed, 'numpy')
        def clf_func(vars):
            return clf_eval(*vars)
        def cbf_func(vars):
            return cbf_eval(*vars)
        
        # Solve
        if i == 0:
            u = np.array([0, 0])
            d = 0
        result = scipy.optimize.minimize(
            cost_func, 
            [*u, d], # [input, delta]
            constraints=[
                {'type': 'ineq', 'fun': clf_func},
                {'type': 'ineq', 'fun': cbf_func},
            ],
        ) 
        u = result.x[:-1]
        d = result.x[-1]
        print(result.success, result.message, result.x)

        # Integrate dynamics
        x += dx_dt(x, u) * (time[i+1] - time[i])

        # Bookkeep
        u_history[i] = u
        x_history[i+1] = x        

    # Visualize
    cmap = plt.cm.viridis
    norm = Normalize(vmin=time[0], vmax=time[-1])
    points = x_history[:,:2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(time)
    lc.set_linewidth(2)
    segments2 = np.concatenate([points, trajectory.reshape(-1, 1, 2)], axis=1)
    lc2 = LineCollection(segments2, color='red', alpha=0.4)
    lc2.set_linewidth(2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.add_collection(lc2)
    cb = plt.colorbar(lc, ax=ax)
    cb.set_label('Time')
    # Create a circle patch
    import matplotlib.patches as patches
    circle = patches.Circle(pos_obs, radius=rad_obs, edgecolor='b', facecolor='none', linewidth=2)

    # Add the circle to the plot
    ax.add_patch(circle)
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
    trajectory = np.linspace(np.zeros(2), np.ones(2), num=n)

    # Main
    example(
        time,
        trajectory,
    )