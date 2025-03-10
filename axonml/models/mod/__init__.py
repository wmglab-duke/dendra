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
    Load mechanism modules from the specified file paths.
    This function dynamically imports Python files from the given directories,
    extracting modules that are expected to contain mechanism implementations.
    Each mechanism is assumed to have a class with the same name as the file.
    
    Parameters
    ----------
    *paths : str
        Variable number of directory paths where mechanism files are located.
    
    Returns
    -------
    MechanismContainer
        A container object that holds all successfully loaded mechanisms.
        The mechanisms are accessible as attributes of the container,
        with attribute names corresponding to the module names.
    
    Notes
    -----
    - Files must have a .py extension and not be named "__init__.py"
    - Each file should define a class with the same name as the file itself
    - The function assumes the module structure follows the convention where
      the class name matches the file name
    
    Examples
    --------
    >>> mechanisms = load_mechanisms('/path/to/mechanisms', '/another/path')
    >>> my_mechanism = mechanisms.mechanism_name
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
    
    mechanisms = MechanismContainer(**mechanisms)

    return mechanisms


class MechanismContainer:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def available(self):
        return list(self.__dict__.keys())
