func_string = """
def new_getattr(obj):
    return obj.{k}
"""


def make_getattr(k):
    exec(func_string.format(k=k))
    return locals()["new_getattr"]
