import matplotlib.pyplot as plt
import numpy as np

from Cholesky import *
from model import *
from PHSkernel_se_New import *
from scipy.integrate import odeint


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

    # dX = dX.T.flatten()
    # Output data / corrupted by some noise
    # dX = dX + 1 * np.random.randn(*dX.shape)

    #dX = dX

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

    # # learning of parameters
    # model.optimize_Hyp()
    # print(f'sd: {model.get_Hyp_sd()}')
    # print(f'l: {model.get_Hyp_l()}')
    # print(f'JR: {model.get_Hyp_JR()}')
    # X = np.loadtxt('X.csv', delimiter=',')
    # dX = np.loadtxt('dX.csv', delimiter=',')
    # dX = dX.T.reshape((X.shape))
    # print(X.shape, dX.shape)
    X = X + 0.2 * np.random.randn(*X.shape)
    model = Model(t_span=np.arange(0, Result_num_data, 0.1), X=X, dX=None, G=G, u=u)

    # ---testing
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, Result_num_data, 0.1), dX[0, :], color='red', label='Ground Truth dX', linewidth=1)
    plt.plot(np.arange(0, Result_num_data, 0.1), model.get_dX()[0, :], color='blue', label='PyGpPhs computed dX', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, Result_num_data, 0.1), dX[1, :], color='red', label='Ground Truth dX', linewidth=1)
    plt.plot(np.arange(0, Result_num_data, 0.1), model.get_dX()[1, :], color='blue', label='PyGpPhs computed dX', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.legend()
    # plt.show()


    #--- testing end

    model.train()
    JR_learned = model.get_Hyp_JR()
    JRX = np.kron(np.eye(n_data), JR_learned)

    # test data and prediction
    alpha = model.get_alpha()
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
    surf2 = ax.plot_surface(X_test, Y_test, np.reshape(pred, n_test), color='gold',linewidth=1, label='PyGpPhs Hamiltonian', alpha=1)
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

    #x_gp = odeint(ode_fun_gp, x0, t_span, args=(JR, JRX, alpha, G, u_test, model.get_Hyp(), X))
    x_gp = model.pred_trajectory(t_span, x0, G, u_test)
    x_org = odeint(ode_fun, x0, t_span, args=(JR_GT, H, G, u_test))

    plt.subplot(2, 1, 1)
    plt.plot(t_span, x_org[:, 0], color='red', label='Ground Truth', linewidth=1)
    plt.plot(t_span, x_gp[:, 0], color='blue', label='PyGpPhs', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_span, x_org[:, 1], color='red', label='Ground Truth', linewidth=1)
    plt.plot(t_span, x_gp[:, 1], color='blue', label='PyGpPhs', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Momentum (Kg m/s)')
    plt.legend()
    plt.show()

    #compute MSE

    # sum_of_err = 0
    # for i in range(len(t_span)):
    #     sum_of_err += np.linalg.norm(x_org[i, :] - x_gp[i, :]) ** 2
    # mse = sum_of_err / len(t_span)
    # print(f"MSE is {mse} for Num data {number_of_data * 10 + 1}")
    # return mse


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    number_of_data = 5
    Prep_num_data = number_of_data + 0.01
    Result_num_data = number_of_data + 0.1
    main()
    # mse_lst = []
    # num_lst = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # number_of_data = 5
    # repetition = 5
    # for number_of_data in [1]:
    #     mse_for_N = []
    #     for i in range(repetition):
    #         Prep_num_data = number_of_data + 0.01
    #         Result_num_data = number_of_data + 0.1
    #         mse_for_N.append(main())
    #     mse_lst.append(np.mean(mse_for_N))
    #     print(f"avg mse for {number_of_data*10+1} is {mse_lst[-1]}")
    # X = np.array([i * 10 + 1 for i in num_lst])
    # Y = np.array([52.04702218278461, 9.961184101181649, 8.01663169228639, 7.583403811789016, 8.480281431820899, 8.201275163011973, 5.024106726485076, 4.998991007224392, 0.1719925580275049, 0.3434543871015817, 0.28])
    # cubic_interpolation_model = interp1d(X, Y, kind="cubic")
    #
    # # Plotting the Graph
    # X_ = np.linspace(X.min(), X.max(), 500)
    # Y_ = cubic_interpolation_model(X_)
    # plt.plot(X_, Y_, label='MSE', linewidth=1)
    # plt.xlabel('Number of training data')
    # plt.ylabel('Mean Squared Error (MSE)')
    # plt.show()

