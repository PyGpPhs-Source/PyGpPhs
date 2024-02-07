from setuptools import setup, Extension, find_packages

# Compile *mysum.cpp* into a shared library
setup(
    name='PyGpPhs_py',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'torch>=1.8.0',
        'scipy>=1.4.0',
        'gpytorch>=1.5.0',
    ],
    ext_modules=[Extension('Command_center', ['PHSkernel_se_CPP.cpp'],
                           extra_compile_args=['-std=c++14'],  # or '-std=c++11' if needed
                           include_dirs=['/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3'],
                           language='c++')],
)