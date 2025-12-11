# Documentation

To build the sphinx documentation and run locally:

```
python -m pip install '.[doc]'  # from the repo root
make html
cd _build/html
python -m http.server
```
