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
import re
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
# Cache unchanged notebooks for ordinary local builds. CI overrides this with
# ``force`` so every documentation run validates every executable notebook.
nb_execution_mode = "cache"
nb_execution_allow_errors = False
nb_execution_raise_on_error = True
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

# Public and local builds link to the public GitHub source. Internal GitLab CI
# keeps its source link on the canonical development repository.
if gitlab_project_url := os.environ.get("CI_PROJECT_URL"):
    html_theme_options["gitlab_url"] = gitlab_project_url
else:
    github_server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    github_repository = os.environ.get("GITHUB_REPOSITORY", "wmglab-duke/dendra")
    html_theme_options["github_url"] = f"{github_server}/{github_repository}"

html_css_files = ["custom.css"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
add_module_names = False
autodoc_member_order = "bysource"

autoclass_content = "both"
autodoc_inherit_docstrings = False


def _signature_parts(body):
    """Split a displayed signature without splitting defaults or annotations."""
    parts, start, depth, quote, escaped = [], 0, 0, None, False
    for index, char in enumerate(body):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(body[start:index].strip())
            start = index + 1
    parts.append(body[start:].strip())
    return [part for part in parts if part]


def public_api_signature(app, what, name, obj, options, signature, return_annotation):
    """Display public arguments and stable omission defaults in the reference."""
    if signature is None:
        return None
    signature = re.sub(r"<[\w.]+ object(?: at 0x[0-9a-fA-F]+)?>", "...", signature)
    try:
        parameters = inspect.signature(obj).parameters.values()
    except (TypeError, ValueError):
        return signature, return_annotation
    private = {
        parameter.name for parameter in parameters if parameter.name.startswith("_")
    }
    if any(
        parameter.name in private
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters
    ):
        return "", None
    if private and signature.startswith("(") and signature.endswith(")"):
        parts = _signature_parts(signature[1:-1])
        parts = [
            part
            for part in parts
            if re.split(r"\s*[:=]", part, maxsplit=1)[0] not in private
        ]
        if "*" in parts:
            index = parts.index("*")
            if all(part.startswith("**") for part in parts[index + 1 :]):
                parts.pop(index)
        signature = "(" + ", ".join(parts) + ")"
    return signature, return_annotation


_viewcode_backlink = re.compile(
    r'(?P<start><div class="viewcode-block" id="(?P<object>[\w.]+)">\s*'
    r'<a class="viewcode-back" href="(?P<path>[^"#]+)#)(?P<anchor>[\w.]+)"'
)


def repair_viewcode_backlinks(app, pagename, _templatename, context, _doctree):
    """Resolve source backlinks for modules with documented re-exports."""
    if not pagename.startswith("_modules/dendra/"):
        return
    body = context.get("body", "")
    if "viewcode-back" not in body:
        return

    # Viewcode uses one reference prefix per source module. A re-export can
    # give other objects in that module a prefix absent from the API reference.
    module = pagename.removeprefix("_modules/").replace("/", ".")
    anchors = {
        anchor: docname
        for _, _, _, docname, anchor, _ in app.env.get_domain("py").get_objects()
    }

    def points_to(anchor, path):
        docname = anchors.get(anchor)
        return (
            docname is not None
            and app.builder.get_relative_uri(pagename, docname) == path
        )

    def repair(match):
        canonical = f"{module}.{match['object']}"
        path = match["path"]
        if not points_to(match["anchor"], path) and points_to(canonical, path):
            return f'{match["start"]}{canonical}"'
        return match[0]

    context["body"] = _viewcode_backlink.sub(repair, body)


def setup(app):
    app.connect("autodoc-process-signature", public_api_signature)
    app.connect("html-page-context", repair_viewcode_backlinks)
