from setuptools import setup

setup(
    install_requires=[
        "numpy",
        "sympy >= 1.2",
        "torch>=2.12.0",
        "scipy",
        "h5py",
        "tqdm",
        "natsort",
        "dask",
        "networkx",
        "pandas",
        "neuron",
        "matplotlib",
        "ninja",
    ],
    extras_require={
        "solvers": [
            "dendra-solvers>=0.3.1",
        ],
        "jupyter": [
            "ipympl >= 0.9.5",
        ],
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
            "shibuya",
        ],
        "dev": [
            "pre-commit",
            "pytest",
            "pytest-cov",
            "hypothesis",
            "pytest-xdist",
        ],
    },
)
