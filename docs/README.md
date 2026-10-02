# Documentation

The public documentation is published at
<https://wmglab-duke.github.io/dendra/> from the GitHub `main` branch.

From the repository root, install the documentation dependencies and the
companion model library, execute every notebook, and serve the result locally:

```sh
python -m pip install ".[doc,solvers]"
DENDRA_MODELS_REVISION="$(cat docs/dendra-models-revision.txt)"
python -m pip install --no-deps \
  "dendra-models @ git+https://github.com/wmglab-duke/dendra-models.git@${DENDRA_MODELS_REVISION}"
sphinx-build -E -W --keep-going \
  -D nb_execution_mode=force \
  -D nb_execution_allow_errors=0 \
  -D nb_execution_raise_on_error=1 \
  -b html docs docs/_build/html
python -m http.server --directory docs/_build/html
```

This matches the public documentation CI contract: every executable notebook
is run, and any cell error fails the build. The recorded revisions pin the
corresponding internal GitLab source commit and filtered public GitHub commit
used by the internal and public documentation workflows. The multicontact
animation tutorial also requires the `ffmpeg` executable on `PATH`; install it
with your operating system's package manager before running the full build.

For a faster prose-only preview that uses stored notebook outputs, replace the
Sphinx command with:

```sh
sphinx-build -W --keep-going -D nb_execution_mode=off \
  -b html docs docs/_build/html
```
