# PHS_simulation
# Author: Tommy Li
# Date: Jul.10, 2023
# Description: The Python interface for the kernel function. This is the interface that provides
#               abstraction to the C code. This function is meant to be imported by other files
#               that need to call the covariance kernel. The C code is compile to the shared lib
#               file format(.so), and this function needs "PHSKernel_se.so" file in the directory.
import ctypes
import numpy as np
import glob
import platform


def PHS_kernel_new(A, B, hyp_sd, hyp_l, d1, JR=None):
    # Load the shared library
    if platform.system() == 'Windows':
        libfile = "PyGpPhs/Extensions/exec/kernel_64bit.dll"
        mylib = ctypes.cdll.LoadLibrary(libfile)
        command_center_proto = ctypes.WINFUNCTYPE(
            ctypes.c_void_p,  # this is the return type!
            ctypes.POINTER(ctypes.c_double),  # X
            ctypes.POINTER(ctypes.c_double),  # Y
            ctypes.c_double,  # sd
            ctypes.POINTER(ctypes.c_double),  # l
            ctypes.c_int,  # d1
            ctypes.c_int,  # rowA
            ctypes.c_int,  # colA
            ctypes.c_int,  # rowB
            ctypes.c_int,  # colB
            ctypes.c_int,  # rowL
            ctypes.POINTER(ctypes.c_double),  # result param
            ctypes.POINTER(ctypes.c_double)  # JR matrix
        )
        Command_center = command_center_proto(("Command_center", mylib), )

    else:
        libfile = glob.glob('PyGpPhs/Extensions/exec/Command_center.cpython-39-darwin.so')[0]
        mylib = ctypes.CDLL(libfile)

        # Define the argument types for the C function
        mylib.Command_center.argtypes = (
            ctypes.POINTER(ctypes.c_double),  # X
            ctypes.POINTER(ctypes.c_double),  # Y
            ctypes.c_double,  # sd
            ctypes.POINTER(ctypes.c_double),  # l
            ctypes.c_int,  # d1
            ctypes.c_int,  # rowA
            ctypes.c_int,  # colA
            ctypes.c_int,  # rowB
            ctypes.c_int,  # colB
            ctypes.c_int,  # rowL
            ctypes.POINTER(ctypes.c_double),  # result param
            ctypes.POINTER(ctypes.c_double)  # JR matrix
        )
        Command_center = mylib.Command_center
        # Define the return type
        Command_center.restype = None

    A = A.T
    B = B.T
    rowA = A.shape[0]
    colA = A.shape[1] if A.ndim != 1 else 1
    rowB = B.shape[0]
    colB = B.shape[1] if B.ndim != 1 else 1
    rowL = len(hyp_l)

    # type conversion to C
    # Convert input matrices to appropriate types
    A_ptr = np.ascontiguousarray(np.array(A), dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    B_ptr = np.ascontiguousarray(np.array(B), dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Convert hyp_l to appropriate type
    hyp_l_ptr = np.ascontiguousarray(np.array(hyp_l), dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if d1 == 2:
        # Create the result matrix
        dims = (colA * rowB, colA * rowA)
        outMatrix = np.zeros(dims, dtype=np.float64)

        # convert C to C_ptr
        C_ptr = np.ascontiguousarray(outMatrix, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if JR is None:
            # Call the C function with the result matrix
            JR_ID = np.identity(colA)
            JR_ptr = np.ascontiguousarray(JR_ID, dtype=np.float64).ctypes.data_as(
                ctypes.POINTER(ctypes.c_double))
            Command_center(A_ptr, B_ptr, hyp_sd, hyp_l_ptr, d1, rowA, colA, rowB, colB, rowL, C_ptr, JR_ptr)
        else:
            JR = JR.T
            JR_ptr = np.ascontiguousarray(np.array(JR), dtype=np.float64).ctypes.data_as(
                ctypes.POINTER(ctypes.c_double))
            Command_center(A_ptr, B_ptr, hyp_sd, hyp_l_ptr, d1, rowA, colA, rowB, colB, rowL, C_ptr, JR_ptr)
    else:
        dummy_JR = np.array([0])
        d_JR_ptr = np.ascontiguousarray(dummy_JR, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if d1 == 1:
            # Create the result matrix
            dims = (rowB, colA * rowA)
            outMatrix = np.empty(dims, dtype=np.float64)

            # convert C to C_ptr
            C_ptr = np.ascontiguousarray(outMatrix, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            # Call the C function with the result matrix
            Command_center(A_ptr, B_ptr, hyp_sd, hyp_l_ptr, d1, rowA, colA, rowB, colB, rowL, C_ptr, d_JR_ptr)
            outMatrix = outMatrix.T
        elif d1 == 0:
            dims = (rowB, rowA)
            outMatrix = np.empty(dims, dtype=np.float64)

            # convert C to C_ptr
            C_ptr = np.ascontiguousarray(outMatrix, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            # Call the C function with the result matrix
            Command_center(A_ptr, B_ptr, hyp_sd, hyp_l_ptr, d1, rowA, colA, rowB, colB, rowL, C_ptr, d_JR_ptr)
            outMatrix = outMatrix.T
        else:
            return;

    return outMatrix
