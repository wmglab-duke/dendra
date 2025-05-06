import textwrap

def indent(text, level=0):
    return textwrap.indent(text, " " * (4 * level))