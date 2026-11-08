# Documentation

The public documentation is published at
<https://wmglab-duke.github.io/dendra/> from the GitHub `main` branch.

From the repository root, install the documentation dependencies and the
companion model library, execute every notebook, and serve the result locally:

```sh
python -m pip install ".[doc,solvers]" ipywidgets
DENDRA_MODELS_REVISION="$(cat docs/dendra-models-revision.txt)"
python -m pip install --no-deps \
  "dendra-models @ git+https://github.com/wmglab-duke/dendra-models.git@${DENDRA_MODELS_REVISION}"
MPLBACKEND=module://matplotlib_inline.backend_inline \
NEURON_MODULE_OPTIONS=-nogui \
sphinx-build -E -W --keep-going \
  -D nb_execution_mode=force \
  -D nb_execution_allow_errors=0 \
  -D nb_execution_raise_on_error=1 \
  -b html docs docs/_build/html
python docs/check_notebook_outputs.py docs/_build/html
python -m http.server --directory docs/_build/html
```

This matches the public documentation CI contract. Every executable notebook
runs, and any cell error fails the build. The inline backend captures figures
for the static pages, and the output check verifies the tutorial plots and
videos. The recorded revisions pin the Dendra Models 0.3.0 source on GitLab
and its filtered counterpart on GitHub for the internal and public
documentation workflows. The multicontact animation tutorial also requires
the `ffmpeg` executable on `PATH`. Install it with your operating system's
package manager before running the full build.

For a faster prose-only preview that uses stored notebook outputs, replace the
Sphinx command with:

```sh
sphinx-build -W --keep-going -D nb_execution_mode=off \
  -b html docs docs/_build/html
```
