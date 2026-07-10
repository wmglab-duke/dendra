# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import inspect
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "Dendra"
copyright = "2025, WMG Lab (Duke University)"
author = "Minhaj Hussain"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "myst_nb",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pytorch": ("https://pytorch.org/docs/stable", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".myst": "myst-nb",
    ".ipynb": "myst-nb",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "__MACOSX",
    "**/.ipynb_checkpoints",
    "**/__pycache__",
]

# Myst-NB
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "substitution",
]
nb_execution_timeout = 600
nb_execution_mode = "cache"
_here = Path(__file__).resolve().parent
_root = _here.parent
myst_substitutions = {
    "license_text": (_root / "LICENSE.md").read_text(encoding="utf-8"),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.

html_static_path = ["_static"]

html_title = ""
html_logo = "_static/logo-light.png"
html_theme = "shibuya"
html_theme_options = {
    "light_logo": "logo-light.png",  # file is _static/logo-light.png
    "dark_logo": "logo-dark.png",  # file is _static/logo-dark.png
    "accent_color": "cyan",
    "gitlab_url": "https://gitlab.oit.duke.edu/mah148/dendra",
}

html_css_files = ["custom.css"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
add_module_names = False
autodoc_member_order = "bysource"

autoclass_content = "class"
autodoc_inherit_docstrings = False


def skip_inplace_methods(app, what, name, obj, skip, options):
    # Only touch class members
    if what == "class":
        # Skip methods like `initialize_`, `fit_`, etc.
        # but do NOT skip dunder methods like __init__
        if name.endswith("_") and not name.endswith("__"):
            # Optionally also ensure it's a routine (method/function)
            if inspect.isroutine(obj):
                return True  # tell Sphinx to skip this member

    # Fall back to the default behavior for everything else
    return None  # or `return skip` is also acceptable


def setup(app):
    app.connect("autodoc-skip-member", skip_inplace_methods)
