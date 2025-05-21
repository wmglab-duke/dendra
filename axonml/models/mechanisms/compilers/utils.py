import re
import inspect
import string
import random
import textwrap

def multiply_return_value(code_string, multiplier_expr: str) -> str:
    """
    Given a function body in a string,
    replace any `return x` statement with `return <multiplier_expr> * x`.
    """
    # Pattern captures:
    # (1) the word 'return'
    # (2) optional whitespace
    # (3) the return expression (grouped as (.+) to capture it)
    pattern = r"(return)\s+(.+)"

    # Use an f-string to insert the multiplier expression
    # before whatever was captured in group 2.
    replacement = rf"return {multiplier_expr} * \2"

    # Perform the substitution.
    new_code = re.sub(pattern, replacement, code_string)
    return new_code


def load(m, attr: str, cls: type):
    """
    Attempts to retrieve an attribute from an object, falling back to a class attribute if not found.

    Parameters
    ----------
    m : object
        The object from which to retrieve the attribute.
    attr : str
        The name of the attribute to retrieve.
    cls : type
        The class from which to retrieve the attribute if it is not found in the object.

    Returns
    -------
    Any
        The value of the attribute from the object or the class.
    """
    try:
        return getattr(m, attr)
    except AttributeError:
        return getattr(cls, attr)


def randomword(length: int) -> str:
    """
    Generate a random word of a given length.

    Parameters
    ----------
    length : int
        The length of the random word to generate.

    Returns
    -------
    str
        A randomly generated word consisting of lowercase letters.
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def indent(text: str, level=0) -> str:
    """
    Indents each line of the given text by a specified number of levels.

    Parameters
    ----------
    text : str
        The text to be indented.
    level : int, optional
        The number of indentation levels. Each level corresponds to 4 spaces. Default is 0.

    Returns
    -------
    str
        The indented text.
    """
    return textwrap.indent(text, " " * (4 * level))


def get_function_body_as_str(func):
    """
    Extracts the body of a function as a string.

    Parameters
    ----------
    func : function
        The function whose body is to be extracted.

    Returns
    -------
    str
        The body of the function as a string.
    """
    source_lines = inspect.getsourcelines(func)[0]  # Get source code as lines
    body_lines = source_lines[1:]  # Skip the first line (def line)
    body = "".join(body_lines)  # Combine into a single string
    return body


def get_function_as_str(func):
    """
    Extracts the entire function as a string, including its definition.

    Parameters
    ----------
    func : function
        The function to be extracted.

    Returns
    -------
    str
        The entire function as a string.
    """
    source_lines = inspect.getsourcelines(func)[0]  # Get source code as lines
    body = "".join(source_lines)  # Combine into a single string
    return body
