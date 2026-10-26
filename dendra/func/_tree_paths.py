"""Sequential lookup and in-place insertion for nested explicit frames."""


def _get_path(tree, path):
    """Read each path segment in order, returning the root for an empty path."""
    value = tree
    for part in path:
        value = value[part]
    return value


def _put_path(tree, path, value):
    """Insert in place, retaining prefix writes if a later access fails."""
    cursor = tree
    for part in path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[path[-1]] = value
