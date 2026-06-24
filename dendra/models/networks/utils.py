import keyword

from dendra.utils.dynamic_compilation import compile_generated_function

func_string = """
def new_getattr(obj):
    return obj.{k}
"""


def validate_attr_path(k: str) -> str:
    if not isinstance(k, str):
        raise TypeError(f"attribute path must be str, got {type(k).__name__}")

    k = k.strip()
    parts = k.split(".")

    if not parts or any(part == "" for part in parts):
        raise ValueError(f"invalid attribute path: {k!r}")

    for part in parts:
        if not part.isidentifier() or keyword.iskeyword(part):
            raise ValueError(f"invalid attribute path segment {part!r} in {k!r}")

    return k


def make_getattr(k):
    k = validate_attr_path(k)

    return compile_generated_function(
        func_string.format(k=k),
        func_name="new_getattr",
        filename_prefix=f"dendra.getattr.{k}",
        global_ns={},
    )
