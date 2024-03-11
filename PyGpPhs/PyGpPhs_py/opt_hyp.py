from scipy.optimize import minimize
from PyGpPhs.PyGpPhs_py.PHSkernel_se_New import *
from PyGpPhs.PyGpPhs_py.hyp import Hyp
from PyGpPhs.PyGpPhs_py.Cholesky import *


def opt_hyp(hyp, lb, ub, X, dX):
    hyp_vec = homogenize_arr([hyp.get_SN()[0], hyp.get_SD()[0], (hyp.get_L()[0]), (hyp.get_JRvec()[0])])
    to_opt = homogenize_arr([hyp.get_SN()[1], hyp.get_SD()[1], (hyp.get_L()[1]), (hyp.get_JRvec()[1])])
    hyp_init = [hyp_vec[i] for i in range(len(to_opt)) if to_opt[i]]

    result = minimize(wrapper, hyp_init, args=(hyp_vec, to_opt, X, dX), method='Nelder-Mead', bounds=list(zip(lb, ub)))

    dim = X.shape[0]

    hyp_vec = update_hyp_vec(hyp_vec, to_opt, result.x)
    hyp_out = Hyp()
    hyp_out.set_SN(hyp_vec[0])
    hyp_out.set_SD(hyp_vec[1])
    hyp_out.set_L(hyp_vec[2:3 + dim - 1])
    hyp_out.set_JRvec(np.zeros((dim, dim)))
    tri_idx = np.triu_indices(dim)
    hyp_out_JR_values = hyp_vec[2 + dim:]
    hyp_out_JR = np.zeros((dim, dim))
    hyp_out_JR[tri_idx] = hyp_out_JR_values
    symmetric_JR = hyp_out_JR - hyp_out_JR.T + np.diag(np.diag(hyp_out_JR))
    hyp_out.set_JRvec(symmetric_JR)

    return hyp_out


def homogenize_arr(Arr):
    Y = []
    for arr in Arr:
        if isinstance(arr, int):
            Y.append(int(arr))
        else:
            for a in arr:
                Y.append(int(a))
    return Y


def wrapper(x, hyp_vec, to_opt, X, dX):
    dim = X.shape[0]
    JR = np.zeros((dim, dim))
    hyp = np.zeros((2 + dim))

    hyp_vec = update_hyp_vec(hyp_vec, to_opt, x)

    hyp[:] = hyp_vec[:3 + dim - 1]
    JR[np.triu_indices(dim)] = hyp_vec[3 + dim - 1:]
    JR = JR - JR.T + np.diag(np.diag(JR))

    out = logmaglik(hyp, JR, X, dX)
    return -out


def logmaglik(hyp, JR, X, dX):
    n = dX.shape[0]
    n_data = X.shape[1]
    K = PHS_kernel_new(X, X, hyp[1], hyp[2:], 2, JR)

    L = Cholesky_decomp(K + (hyp[0] ** 2) * np.eye(n))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, dX))

    out = -0.5 * dX.T.dot(alpha) - np.sum(np.log(np.diag(L))) - n / 2.0 * np.log(2.0 * np.pi)
    return out


def update_hyp_vec(hyp_arr, opt_arr, x_arr):
    indeX = 0
    for i in range(len(hyp_arr[:])):
        if opt_arr[i]:
            hyp_arr[i] = x_arr[indeX]
            indeX += 1
    return hyp_arr
