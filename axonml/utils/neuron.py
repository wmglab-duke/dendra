from neuron import h


def _parent_section(sec):
    """
    Return the parent Section of `sec`, or None if `sec` is a root.

    Uses Section.parentseg() when available (returns a Segment or None);
    otherwise falls back to SectionRef.has_parent()/parent (must guard
    has_parent() to avoid an execution error). :contentReference[oaicite:1]{index=1}
    """
    if hasattr(sec, "parentseg"):
        pseg = sec.parentseg()
        return None if pseg is None else pseg.sec

    sref = h.SectionRef(sec=sec)
    return sref.parent if sref.has_parent() else None


def _chain_to_root(sec):
    """List of sections [sec, parent, parent, ..., root]."""
    chain = []
    while sec is not None:
        chain.append(sec)
        sec = _parent_section(sec)
    return chain


def _sec_key(sec):
    """
    Stable identifier for dict/set membership.
    Section objects are usually usable directly, but name() is robust.
    """
    return sec.name()


def path_sections(a, b):
    """
    Unique simple section-path from section a to section b (inclusive),
    returned as a Python list ordered [a, ..., b].

    Returns None if a and b are disconnected (different trees).
    """
    a2r = _chain_to_root(a)
    b2r = _chain_to_root(b)

    # Map b-ancestors to index in b2r for O(1) LCA lookup
    b_pos = {_sec_key(sec): i for i, sec in enumerate(b2r)}

    lca_i = lca_j = None
    for i, sec in enumerate(a2r):
        j = b_pos.get(_sec_key(sec))
        if j is not None:
            lca_i, lca_j = i, j
            break

    if lca_i is None:
        return None  # disconnected

    # a -> ... -> LCA
    up = a2r[: lca_i + 1]
    # LCA's child -> ... -> b (reverse b->...->LCA, excluding LCA)
    down = list(reversed(b2r[:lca_j]))

    return up + down


def path_via(a, b, c, *, return_sectionlist=False):
    """
    Sections on a simple path from a to b that passes through c
    (no repeated sections). Returns None if no such simple path exists.

    If return_sectionlist=True, returns a NEURON h.SectionList built from
    the resulting iterable. :contentReference[oaicite:2]{index=2}
    """
    p = path_sections(a, b)
    if p is None:
        return None

    ck = _sec_key(c)
    if all(_sec_key(sec) != ck for sec in p):
        # In a tree, this means there is no simple a->b path that goes through c.
        return None

    if return_sectionlist:
        return h.SectionList(
            p
        )  # SectionList can be constructed from a python iterable. :contentReference[oaicite:3]{index=3}
    return p
