import numpy as np
import sympy
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.patches as patches


def calculate_distance(trajectory):
    distances = [np.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory) - 1)]
    return np.sum(distances)


# -- Main -- #

def follow_trajectory(
    time,
    trajectory,
    dynamics='double_integrator'
):
    """Dynamical system has affine form dx_dt = f(x) + g(x)u
    """
    n, _ = trajectory.shape

    if dynamics == 'double_integrator':
        # X/Y Double integrator
        state = sympy.Matrix(sympy.symbols('x y vx vy'))
        input = sympy.Matrix(sympy.symbols('ux uy'))
        f = state[2:,:].col_join(sympy.zeros(2,1))
        g = sympy.zeros(2).col_join(sympy.eye(2))

        kp = 0.2
        clf_const = 100.
        cbf_const_1 = 0.1
        cbf_const_2 = 0.1

        u_lim = 0.001
        u_lim_exprs = [
            input[0] + u_lim, 
            -input[0] + u_lim,
            input[1] + u_lim, 
            -input[1] + u_lim,
        ]

    elif dynamics == 'unicycle':
        # Unicycle model
        speed = calculate_distance(trajectory) / (np.max(time) - np.min(time))
        state = sympy.Matrix(sympy.symbols('x y theta'))
        input = sympy.Matrix([sympy.symbols('omega')])
        f = sympy.Matrix([
            [speed * sympy.cos(state[2])],
            [speed * sympy.sin(state[2])],
            [0],
        ])
        g = sympy.Matrix([
            [0],
            [0],
            [1],
        ])

        kp = 20.
        clf_const = 1.
        cbf_const_1 = 0.1
        cbf_const_2 = 0.1

        u_lim = np.pi/36
        u_lim_exprs = [
            input[0] + u_lim, 
            -input[0] + u_lim,
        ]
    
    else:
        raise NotImplementedError(f"Dynamics {dynamics} not implemented")
    
    # Lambdify dynamics
    f_eval = sympy.utilities.lambdify(state, f, 'numpy')
    g_eval = sympy.utilities.lambdify(state, g, 'numpy')
    dx_dt = lambda x, u: (f_eval(*x) + g_eval(*x).dot(u.reshape(-1,1))).flatten()
    
    # Define Lyapunov function
    state_goal = sympy.Matrix(sympy.symbols('xd yd'))
    K = sympy.diag(1, 1) * kp
    V = (f[:2,:] - K * (state_goal - state[:2,:])).dot(f[:2,:] - K * (state_goal - state[:2,:]))
    
    # Calculate CLF expression, with relaxation
    delta = sympy.symbols('delta')
    clf = - sympy.diff(V, state).dot(f) - sympy.diff(V, state).dot(g * input) - clf_const * V + delta

    # Define barrier function
    pos_obs = sympy.Matrix([0.4, 0.6])
    rad_obs = 0.25
    b = (state[:2,:] - pos_obs).dot(state[:2,:] - pos_obs) - rad_obs ** 2

    # Calculate CBF expressions
    cbf = sympy.diff(b, state).dot(f) + sympy.diff(b, state).dot(g * input) + cbf_const_1 * b
    cbf_2 = sympy.diff(cbf, state).dot(f) + sympy.diff(cbf, state).dot(g * input) + cbf_const_2 * cbf

    # Initialize optimization problem
    W = sympy.eye(len(input))
    p = 1
    cost = (input).dot(W * (input)) + p * delta ** 2
    cost_eval = sympy.utilities.lambdify((*input, delta), cost, 'numpy')
    cost_func = lambda vars: cost_eval(*vars)
    
    # Bookkeeping
    x = np.zeros(len(state))
    x_history = np.zeros((n, len(state)))
    x_history[0] = x.copy()
    u_history = np.zeros((n - 1, len(input)))

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
        u_lim_evals = [sympy.utilities.lambdify((*input, delta), u_lim_expr, 'numpy') for u_lim_expr in u_lim_exprs]
        clf_func = lambda vars: clf_eval(*vars)
        cbf_func = lambda vars: cbf_eval(*vars)
        u_lim_funcs = [lambda vars, u_lim_eval=u_lim_eval: u_lim_eval(*vars) for u_lim_eval in u_lim_evals]
        
        # Solve
        if i == 0:
            u = np.zeros(len(input))
            d = 0
        result = scipy.optimize.minimize(
            cost_func, 
            [*u, d], # [input, delta]
            constraints=[
                {'type': 'ineq', 'fun': clf_func},
                {'type': 'ineq', 'fun': cbf_func},
            ] + [
                {'type': 'ineq', 'fun': u_lim_func} for u_lim_func in u_lim_funcs
            ]
            ,
        )
        u = result.x[:-1]
        d = result.x[-1]
        print(result.success, result.message, result.x)

        # Integrate dynamics
        x += dx_dt(x, u) * (time[i+1] - time[i])

        # Bookkeep
        u_history[i] = u
        x_history[i+1] = x.copy()

    # Visualize
    points = x_history[:,:2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.cm.viridis, norm=Normalize(vmin=time[0], vmax=time[-1]))
    lc.set_array(time)

    fig, ax = plt.subplots()
    ax.add_collection(lc)
    cb = plt.colorbar(lc, ax=ax)
    cb.set_label('Time')

    circle = patches.Circle(pos_obs, radius=rad_obs, edgecolor='b', facecolor='none', linewidth=2)
    ax.add_patch(circle)

    ax.set_title(f"{dynamics} dynamics")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    plt.show()


# -- Executable -- #

if __name__ == '__main__':
    # Script inputs
    n = 100
    time = np.arange(n)
    trajectory = np.linspace(np.zeros(2), np.ones(2), num=n)
    # dynamics = 'double_integrator'
    dynamics = 'unicycle'

    # Main
    follow_trajectory(
        time,
        trajectory,
        dynamics=dynamics,
    )