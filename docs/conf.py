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
import os
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "Dendra"
copyright = "2026, WMG Lab (Duke University)"
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

if os.environ.get("DENDRA_DOCS_DISABLE_INTERSPHINX") == "1":
    # Strict CI validation should fail for warnings in Dendra's documentation,
    # not because an external documentation inventory is temporarily offline.
    intersphinx_mapping = {}
else:
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
        "pytorch": ("https://docs.pytorch.org/docs/stable", None),
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
    "README.md",
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
# Include the underlying exception in CI logs, not only a report-file path.
nb_execution_show_tb = True
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
    "light_logo": "_static/logo-light.png",
    "dark_logo": "_static/logo-dark.png",
    "accent_color": "cyan",
}

# Link to GitHub only in a GitHub-hosted documentation build. During internal
# GitLab development the mirror may still be private, so a hard-coded URL would
# make the source link unusable.
if github_repository := os.environ.get("GITHUB_REPOSITORY"):
    github_server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    html_theme_options["github_url"] = f"{github_server}/{github_repository}"
elif gitlab_project_url := os.environ.get("CI_PROJECT_URL"):
    html_theme_options["gitlab_url"] = gitlab_project_url

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
