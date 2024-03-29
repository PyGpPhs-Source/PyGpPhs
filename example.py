import random

import matplotlib.pyplot as plt
from PyGpPhs.PyGpPhs_py.model import *
from PyGpPhs.PyGpPhs_py.PHSkernel_se_New import *
from scipy.integrate import odeint
import numpy as np


def u(t):
    return 0.1 * np.sin(t)


def G(x):
    return np.array([0, 1])


def H(x):
    return 0.5 * (x[0] ** 2) + 2 * np.cos(x[0]) + 0.5 * (x[1] ** 2) - 1


def dH(x):
    return [x[0] - 2 * np.sin(x[0]), x[1]]


def ode_fun(x, t, JR, H, G, u):
    # Compute state derivatives using PHS form with numerical gradients.
    dim = np.shape(x)[0]
    dH = np.zeros(dim)

    for i in range(dim):
        y = x.copy()
        y[i] = x[i] - 1e-5
        dH[i] = (H(x) - H(y)) / 1e-5

    dx = np.dot(JR, dH) + G(x) * u(t)
    return dx


# this function is called only once for graph production
# Figure1-- function is used for encapsulation of main driver
def graph1(t_span, x_org):
    # Plot the results
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(t_span, x_org[:, 0])
    plt.xlabel('time')
    plt.ylabel('position')
    plt.subplot(2, 1, 2)
    plt.plot(t_span, x_org[:, 1])
    plt.xlabel('time')
    plt.ylabel('momentum')
    # plt.show()


def base_init():
    # base initialization
    k = 10
    R = 0.1
    m = 1
    JR_GT = np.array([[0, 1], [-1, -R]])

    # Define the initial state
    x0 = np.array([5, 0])

    # Define the time span for the simulation
    t_span = np.arange(0, Prep_num_data, 0.01)

    return k, R, m, JR_GT, x0, t_span


def prepare_training_data(x_org, JR_GT, H, G, u):
    # Input data
    t = np.array([0.01 * i for i in range(0, (50 * 100) + 1)])
    X = x_org[0::10, :].T
    t_mod = t[::10]
    n_data = X.shape[1]
    dX = np.zeros((2, n_data))

    # Get the derivatives (for the real system, we would need to use numerical gradients)
    for i in range(n_data):
        dX[:, i] = ode_fun(X[:, i], t_mod[i], JR_GT, H, G, u) - np.dot(G(X[:, i]), u(t_mod[i]))

    return t, X, t_mod, n_data, dX


def get_test_pred(X, JR, hyp, alpha):
    X_test, Y_test = np.meshgrid(np.arange(-4, 4.1, 0.1), np.arange(-4, 4.1, 0.1))
    x_test = np.vstack((Y_test.ravel(), X_test.ravel()))
    n_test = X_test.shape
    JRr = np.kron(np.eye(n_test[0] * n_test[1]), JR)
    k_matrix = PHS_kernel_new(x_test, X, hyp.get_SD(), hyp.get_L(), 1)
    temp = np.reshape(JRr.dot(k_matrix).T, (n_test[0] * n_test[1], len(alpha)))
    pred = temp.dot(alpha)
    return X_test, Y_test, n_test, x_test, JRr, pred


def ode_fun_gp(x, t, JR, JRX, alpha, G, u_tst, hyp, X):
    """
    Differential equation system function for odeint.
    """

    PHS_kernel_res = PHS_kernel_new(x.reshape(len(x), 1), X, hyp.get_SD(), hyp.get_L(), 2)
    PHS_kernel_res.reshape(int(0.5 * PHS_kernel_res.shape[0]), int(2 * PHS_kernel_res.shape[1]))

    # Calculate the weighted sum
    weighted_sum = JR.dot(PHS_kernel_res.T).dot(JRX.T).dot(alpha)

    # Calculate the gradient of the system dynamics
    gradient = G(x)

    # Calculate the external input/control signal
    input_signal = u_tst(t)

    # Calculate the derivative of the state variables
    dxdt = weighted_sum + gradient * input_signal

    return dxdt


def getTrue_hamiltonian(n_test, x_test):
    H_true = np.zeros(n_test[0] * n_test[1])
    for i in range(n_test[0] * n_test[1]):
        H_true[i] = H(x_test[:, i])
    return H_true


def main():
    # base initialization:
    k, R, m, JR_GT, x0, t_span = base_init()

    # Simulate the system using odeint, --Figure 1
    x_org = odeint(ode_fun, x0, t_span, args=(JR_GT, H, G, u))
    graph1(t_span, x_org)

    # prepare for training data
    t, X, t_mod, n_data, dX = prepare_training_data(x_org, JR_GT, H, G, u)

    lb = np.concatenate((1e-6 * np.ones((4, 1)), -10 * np.ones((1, 1))))
    ub = np.concatenate((1000 * np.ones((4, 1)), np.zeros((1, 1))))

    # Add noise to state observation X
    X = X + 0.2 * np.random.randn(*X.shape)
    model = Model(t_span=np.arange(0, Result_num_data, 0.1), X=X, dX=None, G=G, u=u)

    # ---testing
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, Result_num_data, 0.1), dX[0, :], color='red', label='Ground Truth dX', linewidth=1)
    plt.plot(np.arange(0, Result_num_data, 0.1), model.get_dX()[0, :], color='blue', label='PyGpPhs_py computed dX',
             linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, Result_num_data, 0.1), dX[1, :], color='red', label='Ground Truth dX', linewidth=1)
    plt.plot(np.arange(0, Result_num_data, 0.1), model.get_dX()[1, :], color='blue', label='PyGpPhs_py computed dX',
             linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.show()

    # --- testing end

    model.train()

    # test data and prediction
    X_test, Y_test = np.meshgrid(np.arange(-4, 4.1, 0.1), np.arange(-4, 4.1, 0.1))
    x_test = np.vstack((Y_test.ravel(), X_test.ravel()))
    pred = model.pred_H(x_test)
    X_test, Y_test, n_test, x_test, JRr, predooo = get_test_pred(X, JR_GT, model.get_Hyp(), model.get_alpha())
    # true hamiltonian
    H_true = getTrue_hamiltonian(n_test, x_test)

    const = np.min(pred)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf1 = ax.plot_surface(X_test, Y_test, np.reshape(H_true, n_test), rstride=2, cstride=2, color='blue', linewidth=1,
                            label='True Hamiltonian', alpha=1)
    surf2 = ax.plot_surface(X_test, Y_test, np.reshape(pred, n_test), color='gold', linewidth=1,
                            label='PyGpPhs_py Hamiltonian', alpha=1)
    ax.scatter3D(X[0, :], X[1, :], np.zeros(n_data), color='red', marker='o')
    surf1._edgecolors2d = surf1._edgecolor3d
    surf1._facecolors2d = surf1._facecolor3d
    surf2._edgecolors2d = surf2._edgecolor3d
    surf2._facecolors2d = surf2._facecolor3d
    ax.legend()
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Momentum (Kg m/s)')
    ax.set_zlabel('Hamiltonian')

    plt.show()

    # Define the initial state
    x0 = np.array([-3, 0]).T

    # Define the time span for the simulation
    t_span = np.arange(0, 100.01, 0.01)
    u_test = lambda t: 0 * t

    x_gp = model.pred_trajectory(t_span, x0, G, u_test)
    #x_gp = Model.PyGpPhs_pred_X(np.arange(0, Result_num_data, 0.1), X, t_span, x0, G_train=G, u_train=u, G_test=G, u_test=u_test)
    x_org = odeint(ode_fun, x0, t_span, args=(JR_GT, H, G, u_test))

    plt.subplot(2, 1, 1)
    plt.plot(t_span, x_org[:, 0], color='red', label='Ground Truth', linewidth=1)
    plt.plot(t_span, x_gp[:, 0], color='blue', label='PyGpPhs_py', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_span, x_org[:, 1], color='red', label='Ground Truth', linewidth=1)
    plt.plot(t_span, x_gp[:, 1], color='blue', label='PyGpPhs_py', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Momentum (Kg m/s)')
    plt.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random.seed(7)
    number_of_data = 5
    Prep_num_data = number_of_data + 0.01
    Result_num_data = number_of_data + 0.1
    main()
