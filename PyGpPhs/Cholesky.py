import ctypes
import numpy as np
import platform


def Cholesky_decomp(A):
    
    # check if windows
    if platform.system() == 'Windows':
        mylib = ctypes.CDLL("../Extensions/exec/Cholesky_decomp.dll")
    else:
        mylib = ctypes.CDLL("../Extensions/exec/Cholesky_decomp.so")

    # Define the argument types for the C function
    mylib.Cholesky_Decomposition.argtypes = (
        ctypes.POINTER(ctypes.c_double),  # A_copy
        ctypes.POINTER(ctypes.c_double),  # result
        ctypes.c_int  # d1
    )
    mylib.Cholesky_Decomposition.restypes = None
    cholesky = mylib.Cholesky_Decomposition

    result = np.zeros(A.shape, dtype=np.float64)
    A_ptr = np.ascontiguousarray(A, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    R_ptr = np.ascontiguousarray(result, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    cholesky(A_ptr, R_ptr, A.shape[0])
    return result
