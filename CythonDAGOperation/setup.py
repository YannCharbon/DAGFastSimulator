from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Cython DAG Generator simulation',
    ext_modules=cythonize("CythonDAGOperation.pyx"),
)
