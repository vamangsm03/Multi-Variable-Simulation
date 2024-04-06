import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor

rho = 1.0  # density
mu = 1.0  # dynamic viscosity
T_surface = 393.15  # surface temperature
k = 0.026  # thermal conductivity
alpha = 8.7 * 1e-3  # heat transfer coefficient
Cp = 1.0  # specific heat capacity
q = 1.0  # heat flux
Lx = 1.0  # length
A = Lx ** 2  # area
V = 1.0  # volume
Pr = 0.71  # Prandtl number
Ra = 1.0e10  # Rayleigh number

def solve_scm(t_span, y0):
    sol = solve_ivp(williamson_model, t_span, y0, args=(rho, mu, k, alpha, Cp, q, Pr, Ra), method='RK45')
    return sol.t, sol.y[0], sol.y[1]

def williamson_model_gwrm(x, y, rho, mu, k, alpha, Cp, q, Pr, Ra):
    u, T = y
    du_dx = -u / Lx
    dT_dx = -q / (k * Cp)
    return np.array([du_dx, dT_dx])

def boundary_conditions(ya, yb, rho, mu, k, alpha, Cp, q, Pr, Ra):
    u_a, T_a = ya
    u_b, T_b = yb
    return np.array([u_a, T_a - T_surface])

def solve_gwrm(t_span, y0):
    x = np.linspace(0, Lx, 101)

    def fun(x, y):
        return williamson_model_gwrm(x, y, rho, mu, k, alpha, Cp, q, Pr, Ra)

    def bc(ya, yb):
        return boundary_conditions(ya, yb, rho, mu, k, alpha, Cp, q, Pr, Ra)

    sol = solve_bvp(fun, bc, x, y0.T)

    u_gwrm = sol.y[0]
    T_gwrm = sol.y[1]

    return x, u_gwrm, T_gwrm

def train_ann(X, y):
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000)
    model.fit(X, y)
    return model

def predict_ann(model, X):
    return model.predict(X)

def plot_profiles(t, u_scm, T_scm, u_gwrm, T_gwrm, u_ann, T_ann):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t, u_scm, label='SCM')
    ax1.plot(t, u_gwrm, label='GWRM')
    ax1.plot(t, u_ann, label='ANN')
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('Velocity (u)')
    ax1.legend()

    ax2.plot(t, T_scm, label='SCM')
    ax2.plot(t, T_gwrm, label='GWRM')
    ax2.plot(t, T_ann, label='ANN')
    ax2.set_xlabel('Position (x)')
    ax2.set_ylabel('Temperature (T)')
    ax2.legend()

    plt.show()

t_span = (0, Lx)
y0 = [0, T_surface]

t_scm, u_scm, T_scm = solve_scm(t_span, y0)
t_gwrm, u_gwrm, T_gwrm = solve_gwrm(t_span, y0)

X = np.column_stack((t_scm, t_gwrm))
y = np.column_stack((u_scm, u_gwrm, T_scm, T_gwrm))
ann_model = train_ann(X, y)

u_ann = predict_ann(ann_model, X[:, 0])
T_ann = predict_ann(ann_model, X[:, 1])

plot_profiles(t_scm, u_scm, T_scm, u_gwrm, T_gwrm, u_ann, T_ann)
