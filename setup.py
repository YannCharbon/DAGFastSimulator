from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="CythonDAGOperation",
        sources=["./CythonDAGOperation/CythonDAGOperation.pyx"]
    )
]

setup(
    name='Cython DAG Generator simulation',
    ext_modules=cythonize(extensions)
)
