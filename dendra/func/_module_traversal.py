"""Raw registered-module traversal with caller-owned qualification."""


def _raw_module_namespace(module):
    return object.__getattribute__(module, "__dict__")


def _registered_modules(module, *, duplicates=False, namespace=vars):
    """Yield registered paths before reading each module's child registrations.

    Callers can qualify a yielded module before resuming traversal. Canonical
    traversal deduplicates module identities; alias traversal only stops
    ancestor cycles. Namespace access retains the caller's lookup semantics.
    """
    seen = set()

    def visit(path, child, ancestors):
        identity = id(child)
        if identity in ancestors or (not duplicates and identity in seen):
            return
        seen.add(identity)
        yield path, child
        for name, nested in namespace(child)["_modules"].items():
            if nested is not None:
                yield from visit(
                    f"{path}.{name}" if path else name,
                    nested,
                    ancestors | {identity},
                )

    return visit("", module, set())
