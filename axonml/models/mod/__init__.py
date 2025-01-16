import os
import importlib

# Get the directory of the current module
module_dir = os.path.dirname(__file__)

# List all Python files in the directory (excluding __init__.py)
py_files = [
    f for f in os.listdir(module_dir) if f.endswith(".py") and f != "__init__.py"
]

# Import each file and fetch its classes
for file in py_files:
    module_name = file[:-3]  # Remove '.py' extension
    module = importlib.import_module(
        f".{module_name}", package=__name__
    )  # Import as relative module
    class_name = (
        module_name  # Assume class name is the capitalized version of file name
    )
    globals()[class_name] = getattr(module, class_name)  # Add class to global namespace


def load_mechanisms(*paths):
    """
    Load mechanisms from a directory containing Python files.

    Args:
        path (str): Path to the directory containing Python files.

    Returns:
        dict: Dictionary of class names and corresponding classes.
    """
    mechanisms = {}

    for path in paths:
        dirpath = os.path.abspath(path)
        py_files = [
            f for f in os.listdir(dirpath) if f.endswith(".py") and f != "__init__.py"
        ]

        for file in py_files:
            # import file as module
            module_name = os.path.splitext(file)[0]
            file_path = os.path.join(dirpath, file)

            # Create a spec from the file and then import the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                mechanisms[module_name] = getattr(module, module_name)

    return mechanisms
