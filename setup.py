from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = cythonize(
    [
        Extension(
            name="axonml.models.heterogeneous.ops",
            sources=["./axonml/models/heterogeneous/ops.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION"],
        ),
    ],
    language_level=3,
)

setup(
    install_requires=[
        "numpy",
        "sympy >= 1.2",
        "torch >= 2.7.0",
        "scipy",
        "h5py",
        "pytorch_optimizer",
        "tqdm",
        "natsort",
        "dask",
    ],
    extras_require={
        "doc": [
            "jupyter_contrib_nbextensions",
            "notebook <= 6.4.12",
            "traitlets <= 5.9.0",
            "ipython <= 8.9.0",
            "mkdocs",
            "mkdocs-material",
            "markdown-include",
            "mkdocs-redirects",
            "mkdocstrings[python]>=0.18",
            "mike",
            "sphinx",
            "sphinx-autobuild",
            "sphinx_autodoc_typehints",
            "sphinx-math-dollar",
            "myst-nb",
            "jupytext",
            "sphinx-book-theme",
        ],
        "dev": [
            "pre-commit",
            "pytest",
            "pytest-cov",
            "hypothesis",
            "pytest-xdist",
        ]
    },
    ext_modules=ext_modules,
)
