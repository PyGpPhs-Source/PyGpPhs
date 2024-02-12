import gpytorch
import numpy as np
from scipy.integrate import odeint
from PyGpPhs.PyGpPhs_py.hyp import Hyp
from PyGpPhs.PyGpPhs_py.Cholesky import *
from PyGpPhs.PyGpPhs_py.opt_hyp import opt_hyp
from PyGpPhs.PyGpPhs_py.PHSkernel_se_New import *
import torch
from PyGpPhs.PyGpPhs_py.GPdX import GP_Model_dX


class Model:

    def __init__(self, t_span, X, dX=None, G=None, u=None, JRX0=None, hyp=Hyp(),
                 lb=np.concatenate((1e-6 * np.ones((4, 1)), -10 * np.ones((1, 1)))),
                 ub=np.concatenate((1000 * np.ones((4, 1)), np.zeros((1, 1))))):
        self.hyp_ = hyp
        self.lb_ = lb
        self.ub_ = ub
        self.X_ = X
        if dX is None:
            self.set_default_dx(t_span, G, u)
        else:
            self.dX_ = dX
        self.JRX_ = JRX0
        self.alpha_ = None

    def get_dX(self):
        return self.dX_

    def set_default_dx(self, t_span, G, u):
        # We will train 1 GP for each dimenison of the data points
        # assume the column vector is the data observations
        # mean funciton from gp = kernel(T*, T)K^(-1)(T,T)S
        # T is the raw obs time step
        # S = [s1, ..., sn]T
        # T* is the desired time step, which in this case T* = T
        dim = self.X_.shape[0]
        n = self.X_.shape[1]
        dX = []
        pred_X = []
        for i in range(dim):
            X_train_tensor = torch.tensor(self.X_[i, :], requires_grad=True)
            t_span_tensor = torch.tensor(t_span, requires_grad=True)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = GP_Model_dX(t_span_tensor, X_train_tensor, likelihood)
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            losses = []
            training_iter = 250
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * training_iter], gamma=0.1)

            for j in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(t_span_tensor)
                # Calc loss and backprop gradients
                loss = -mll(output, X_train_tensor)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(t_span_tensor), requires_grad=True)
            mu = observed_pred.mean.detach().numpy()
            pred_X.append(mu)

            model.eval()
            likelihood.eval()

            # 1st derivative dX
            Tspan = torch.autograd.Variable(t_span_tensor, requires_grad=True)
            observed_pred = likelihood(model(Tspan))

            y = observed_pred.mean.sum()
            y.backward()
            dydtest_x = Tspan.grad
            dX.append(dydtest_x)

        # dX = np.array(dX)
        # ***Impoartant: if error shows up out of range at line 99, uncomment this
        dX = np.array([tensor.detach().numpy() for tensor in dX])

        # subtract Gu
        for i in range(dX.shape[1]):
            dX[:, i] -= np.dot(G(self.X_[:, i]), u(t_span[i]))

        self.dX_ = np.array(dX)

    def get_Hyp(self):
        return self.hyp_

    def get_Hyp_sd(self):
        return self.hyp_.get_SD()

    def get_Hyp_sn(self):
        return self.hyp_.get_SN()

    def get_Hyp_l(self):
        return self.hyp_.get_L()

    def get_Hyp_JR(self):
        return self.hyp_.get_JRvec()

    def get_alpha(self):
        return self.alpha_

    def optimize_Hyp(self):
        self.hyp_ = opt_hyp(self.hyp_, self.lb_, self.ub_, self.X_, self.dX_.T.flatten())

    def train(self, x0=None, t_span=None, JR=None):
        if x0 is not None:
            self.X_ = x0
        self.optimize_Hyp()
        K = PHS_kernel_new(self.X_, self.X_, self.get_Hyp_sd(), self.get_Hyp_l(), 2, self.get_Hyp_JR())
        L = Cholesky_decomp(K + (self.get_Hyp_sn() ** 2) * np.eye(K.shape[0]))
        self.alpha_ = np.linalg.solve(L.T, np.linalg.solve(L, self.dX_.T.flatten()))
        return {'sn': self.get_Hyp_sn(), 'sd': self.get_Hyp_sd(), 'l': self.get_Hyp_l(), 'JR': self.get_Hyp_JR()}

    def pred_H(self, x_test):
        if x_test is None:
            raise Exception('must provide X_test data')
        # x_test = np.vstack((Y_test.ravel(), X_test.ravel()))

        JRr = np.kron(np.eye(x_test.shape[1]), self.get_Hyp_JR())
        k_matrix = PHS_kernel_new(x_test, self.X_, self.get_Hyp_sd(), self.get_Hyp_l(), 1)
        temp = np.reshape(JRr.dot(k_matrix).T, (x_test.shape[1], len(self.get_alpha())))
        predResult = temp.dot(self.get_alpha())
        return predResult - np.min(predResult)

    def pred_trajectory(self, t_span, x0, G, u=lambda t: 0 * t):
        JR = self.get_Hyp_JR()
        n_data = self.X_.shape[1]
        JRX = np.kron(np.eye(n_data), JR)
        x_gp = odeint(self.__ode_fun_gp, x0, t_span, args=(JR, JRX, self.alpha_, G, u, self.get_Hyp(), self.X_))
        return x_gp

    def pred_dx(self, x, t_span, G, u=lambda t: 0 * t):
        return self.__ode_fun_gp(x, t_span, self.get_Hyp_JR(), self.JRX_, self.alpha_, G, u, self.hyp_, self.X_)

    @staticmethod
    def __ode_fun_gp(x, t, JR, JRX, alpha, G, u_tst, hyp, X):
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
