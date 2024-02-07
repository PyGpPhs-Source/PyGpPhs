from setuptools import setup, Extension

# Compile *mysum.cpp* into a shared library
setup(
    ext_modules=[Extension('Command_center', ['PHSkernel_se_CPP.cpp'],
                           extra_compile_args=['-std=c++14'],  # or '-std=c++11' if needed
                           include_dirs=['/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3'],
                           language='c++')],
)