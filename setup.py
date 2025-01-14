from setuptools import setup, Extension

from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize(
    [
        Extension(
            name="axonml.models.heterogeneous.ops",
            sources=["./axonml/models/heterogeneous/ops.pyx"],
            include_dirs=[np.get_include()],
        ),
    ]
)
setup(ext_modules=ext_modules)
