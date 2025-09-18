from setuptools import setup
from Cython.Build import cythonize

setup(
    name = "fast_counter",
    ext_modules=cythonize("fast_counter.pyx"),
    zip_safe=False,
)