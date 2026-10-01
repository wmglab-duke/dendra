# Documentation

From the repository root, build the documentation without executing
notebooks and serve it locally with:

```sh
python -m pip install ".[doc]"
sphinx-build -W --keep-going -D nb_execution_mode=off -b html docs docs/_build/html
python -m http.server --directory docs/_build/html
```

This uses the notebooks' stored outputs and matches the documentation CI build.
Refresh notebook caches only in an environment that also contains
their model and solver dependencies.
