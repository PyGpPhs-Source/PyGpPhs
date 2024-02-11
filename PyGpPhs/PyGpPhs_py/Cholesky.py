import ctypes
import numpy as np
import platform


def Cholesky_decomp(A):
    if platform.system() == 'Darwin':
        mylib = ctypes.CDLL("PyGpPhs/Extensions/exec/Cholesky_decomp.so")
        # Define the argument types for the C function
        mylib.Cholesky_Decomposition.argtypes = (
            ctypes.POINTER(ctypes.c_double),  # A_copy
            ctypes.POINTER(ctypes.c_double),  # result
            ctypes.c_int  # d1
        )
        mylib.Cholesky_Decomposition.restypes = None
        cholesky = mylib.Cholesky_Decomposition
    else:
        libfile = "../Extension/exec/Cholesky_decomp_64bit.dll"
        mylib = ctypes.cdll.LoadLibrary(libfile)
        Cholesky_decomp_proto = ctypes.WINFUNCTYPE(
            ctypes.c_void_p,  # return type
            ctypes.POINTER(ctypes.c_double),  # A_copy
            ctypes.POINTER(ctypes.c_double),  # result
            ctypes.c_int)  # d1
        cholesky = Cholesky_decomp_proto(("Cholesky_Decomposition", mylib), )

    result = np.zeros(A.shape, dtype=np.float64)
    A_ptr = np.ascontiguousarray(A, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    R_ptr = np.ascontiguousarray(result, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    cholesky(A_ptr, R_ptr, A.shape[0])
    return result