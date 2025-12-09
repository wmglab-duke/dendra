# Documentation

To build the sphinx documentation, install the documentation extras and run:
```
python -m pip install '.[doc]'  # from the repo root
make html
cd _build/html
python -m http.server
```
This will find all jupyter notebooks, run them, collect the output, and incorporate them into the documentation.
