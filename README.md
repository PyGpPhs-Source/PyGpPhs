# PyGpPhs Package

The PyGpPhs package is designed for utilizing Gaussian Process for port-Hamiltonian system.

The PyGpPhs toolbox is developed using in Python and C++. The main
structure of the toolbox is written in Python while a
few computationally expensive functions, such as the ker-
nel and Cholesky decomposition functions, are written
in C++ . While two languages are used to develop the
PyGpPhs toolbox, the important notion of encapsulation
and abstraction are largely realized. Utilizing C++ data
types and function calls are handled internally by the
model class, thus leaving no explicit overheads for the users
to consider the C++ internal workings or invoke function
calls.

## Installation

**Requirements**:
- Python>=3.8
- numpy>=1.18.0 
- torch>=1.8.0 
- scipy>=1.4.0 
- gpytorch>=1.5.0

Install GPyTorch using pip or conda:
```bash
pip install pygpphs
conda install pygpphs -c pygpphs
```
More information on installation can be viewed at: 
