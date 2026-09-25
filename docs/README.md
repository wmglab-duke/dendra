# Documentation

To build the sphinx documentation and run locally:

```
python -m pip install '.[doc,solvers]'  # from the repo root; examples use CPU solvers
cd docs
make html
cd _build/html
python -m http.server
```
