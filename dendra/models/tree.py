"""Tree-shaped population models and supporting utilities."""

import keyword
import math

import networkx as nx
import torch
import torch.nn.functional as F

from dendra.models.integrators import dhs

from .core import Population
from .morphology import CompartmentGraph, Morphology

_RESERVED_COMPARTMENT_LABELS = frozenset({"internal_nodes"})


def _is_safe_slice_label(model, name):
    """Return whether a canonical morphology label can become an attribute."""
    return (
        isinstance(name, str)
        and name not in _RESERVED_COMPARTMENT_LABELS
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and not name.startswith("_")
        and (name in model._labels or not hasattr(model, name))
    )


def _register_compartment_graph_labels(model, graph, ordered=None):
    """Install collision-free canonical regions as population Slice labels.

    ``ordered`` may override storage order for a label. Native Section labels
    use it to retain increasing Section coordinate even when a child is attached
    through its ``1`` end and therefore appears reversed in tree traversal.
    """
    label_names = sorted(
        {label for labels in graph.metadata.labels for label in labels}
    )
    for label in label_names:
        indices = (
            tuple(ordered[label])
            if ordered is not None and label in ordered
            else graph.nodes_with_label(label)
        )
        if not indices or not _is_safe_slice_label(model, label):
            continue
        existing = model._labels.get(label)
        candidate = model[:, torch.as_tensor(indices, device=model.device())]
        if existing is not None:
            if ordered is not None and label in ordered:
                candidate.label(label, replace=True)
            # Otherwise legacy Tree labels (soma/dend/apic/axon) already use
            # the same public name and retain their established search order.
            continue
        candidate.label(label)


def _register_canonical_internal_nodes(model, graph):
    """Define material nodes from canonical kind metadata, not node names."""
    indices = tuple(
        node for node, kind in enumerate(graph.metadata.kind) if kind == "compartment"
    )
    if not indices:
        raise ValueError("A Tree compartment graph must contain a material node.")
    model[:, torch.as_tensor(indices, device=model.device())].label(
        "internal_nodes", replace=True
    )


def _normalize_tree_graph(graph):
    """Validate a morphology tree and normalize node labels to tensor indices.

    Canonical graphs whose nodes are already ``0..n-1`` are returned unchanged
    so callers that retain the graph object keep the historical identity
    contract. Other hashable labels are mapped to ``0..n-1`` in deterministic
    NetworkX insertion order on a copy of the graph.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Tree.from_graph requires a NetworkX graph.")
    if not graph.is_directed():
        raise ValueError("Tree morphology must be a directed graph.")
    if graph.is_multigraph():
        raise ValueError("Tree morphology must be a simple directed graph.")
    if graph.number_of_nodes() == 0:
        raise ValueError("Tree morphology must contain at least one node.")
    if any(parent == child for parent, child in graph.edges):
        raise ValueError("Tree morphology must not contain self-loops.")

    multiple_parents = [node for node, degree in graph.in_degree() if int(degree) > 1]
    if multiple_parents:
        raise ValueError(
            "Every Tree morphology node must have at most one parent; "
            f"invalid nodes: {multiple_parents!r}."
        )
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Tree morphology must be acyclic.")
    if not nx.is_weakly_connected(graph):
        raise ValueError("Tree morphology must be connected, not a forest.")

    nodes = list(graph.nodes)
    if set(nodes) == set(range(len(nodes))):
        return graph
    mapping = {node: index for index, node in enumerate(nodes)}
    return nx.relabel_nodes(graph, mapping, copy=True)


def _as_float(value, *, name: str, node=None, default=None) -> float:
    """Convert graph metadata to ``float`` with a useful error message."""
    if value is None:
        if default is not None:
            return float(default)
        where = "" if node is None else f" on graph node {node!r}"
        raise KeyError(f"Missing required morphology attribute {name!r}{where}.")
    return float(value)


def _cylindrical_volume_um3(attrs, *, node=None) -> float:
    """Fallback volume from stylized length/diameter metadata, in µm³."""
    length_um = _as_float(attrs.get("L"), name="L", node=node, default=0.0)
    diam_um = _as_float(attrs.get("diam"), name="diam", node=node, default=0.0)
    if length_um <= 0.0 or diam_um <= 0.0:
        return 0.0
    radius_um = 0.5 * diam_um
    return float(math.pi * radius_um * radius_um * length_um)


def _node_volume_um3(attrs, *, node=None) -> float:
    """Return total compartment volume in µm³.

    Graphs produced by :func:`dendra.models.io.neuron_to_dendra_graph` may carry
    pt3d-aware ``volume``/``volume_um3`` metadata. Older graphs do not, so this
    falls back to the stylized cylinder approximation used for Dendra's native
    one-dimensional axon geometry. Branchpoint nodes with ``L=0`` naturally get
    zero volume.
    """
    if "volume" in attrs:
        return _as_float(attrs.get("volume"), name="volume", node=node)
    if "volume_um3" in attrs:
        return _as_float(attrs.get("volume_um3"), name="volume_um3", node=node)
    return _cylindrical_volume_um3(attrs, node=node)


def _node_domain_volume_um3(
    attrs, domain: str, total_volume: float, *, node=None
) -> float:
    """Return domain-specific volume in µm³.

    ``volume_i`` defaults to the total compartment volume, which is the right
    intracellular default for imported morphologies. ``volume_o`` defaults to
    zero until an extracellular volume model is supplied.
    """
    key = f"volume_{domain}"
    if key in attrs:
        return _as_float(attrs.get(key), name=key, node=node)
    if domain == "i":
        return float(total_volume)
    return 0.0


def _edge_diff_geom_um(graph, parent, child) -> float:
    """Return edge diffusion geometry factor in µm.

    Preferred source is the explicit ``diff_geom_um`` edge attribute written by
    the NEURON-import geometry code. As a compatibility fallback, recover the
    same geometric factor from electrical axial resistance when available:

        diff_geom_um = Ra[Ω·cm] * 1e4 / R_ohm[Ω]

    This fallback is exact only when one effective intracellular resistivity
    applies to the edge. New NEURON-imported graphs should use explicit
    ``diff_geom_um`` instead.
    """
    edge = graph.edges[parent, child]
    if "diff_geom_um" in edge:
        return _as_float(edge.get("diff_geom_um"), name="diff_geom_um")

    R_ohm = edge.get("R_ohm", None)
    if R_ohm is not None and float(R_ohm) != 0.0:
        parent_Ra = graph.nodes[parent].get("Ra", None)
        child_Ra = graph.nodes[child].get("Ra", None)
        if parent_Ra is None and child_Ra is None:
            raise KeyError(
                "Edge is missing diff_geom_um and cannot recover it from R_ohm "
                "because neither endpoint has Ra."
            )
        if parent_Ra is None:
            rhoa = float(child_Ra)
        elif child_Ra is None:
            rhoa = float(parent_Ra)
        else:
            rhoa = 0.5 * (float(parent_Ra) + float(child_Ra))
        return float(rhoa * 1e4 / float(R_ohm))

    # Last-resort stylized geometry fallback: approximate the edge as two
    # half-compartments connected in series. This is mainly for hand-written
    # test graphs and should not be relied on for pt3d morphologies.
    L_edge = _as_float(edge.get("L"), name="L", default=0.0)
    if L_edge <= 0.0:
        return 0.0
    d_parent = _as_float(
        graph.nodes[parent].get("diam"), name="diam", node=parent, default=0.0
    )
    d_child = _as_float(
        graph.nodes[child].get("diam"), name="diam", node=child, default=0.0
    )
    if d_parent <= 0.0 or d_child <= 0.0:
        return 0.0
    a_parent = math.pi * (0.5 * d_parent) ** 2
    a_child = math.pi * (0.5 * d_child) ** 2
    integral = 0.5 * L_edge / a_parent + 0.5 * L_edge / a_child
    return 1.0 / integral if integral > 0.0 else 0.0


def gather_morphology(graph):
    """Extract morphology tensors from a graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        Morphology graph whose nodes provide geometric attributes.

    Returns
    -------
    dict
        Mapping from attribute names to tensors shaped ``(1, n_comp)``.

    Notes
    -----
    In addition to Dendra's historical ``dx``, ``diam``, and coordinate buffers,
    this now registers material-geometry buffers:

    ``volume`` / ``volume_um3``
        Total compartment volume in µm³.

    ``volume_i``
        Intracellular material volume in µm³. Defaults to ``volume``.

    ``volume_o``
        Extracellular material volume in µm³. Defaults to zero until an
        extracellular volume model is supplied.
    """
    L, diam, x, y, z = [], [], [], [], []
    volume, volume_i, volume_o = [], [], []

    for i in range(len(graph.nodes)):
        attrs = graph.nodes[i]
        L.append(attrs.get("L"))
        diam.append(attrs.get("diam"))
        x.append(attrs.get("x", 0.0))
        y.append(attrs.get("y", 0.0))
        z.append(attrs.get("z", 0.0))

        vol = _node_volume_um3(attrs, node=i)
        volume.append(vol)
        volume_i.append(_node_domain_volume_um3(attrs, "i", vol, node=i))
        volume_o.append(_node_domain_volume_um3(attrs, "o", vol, node=i))

    # Graph metadata arrives as Python floats (IEEE-754 binary64).  Preserve
    # those values until Tree.from_graph deliberately converts the gathered
    # buffers to the requested model dtype.  Constructing these tensors with
    # PyTorch's default float32 would irreversibly quantize a float64 Tree.
    volume_t = torch.tensor(volume, dtype=torch.float64).unsqueeze(0)
    return {
        "dx": torch.tensor(L, dtype=torch.float64).unsqueeze(0),
        "diam": torch.tensor(diam, dtype=torch.float64).unsqueeze(0),
        "x": torch.tensor(x, dtype=torch.float64).unsqueeze(0),
        "y": torch.tensor(y, dtype=torch.float64).unsqueeze(0),
        "z": torch.tensor(z, dtype=torch.float64).unsqueeze(0),
        "volume": volume_t,
        "volume_um3": volume_t.clone(),
        "volume_i": torch.tensor(volume_i, dtype=torch.float64).unsqueeze(0),
        "volume_o": torch.tensor(volume_o, dtype=torch.float64).unsqueeze(0),
    }


def gather_diffusion_edges(graph):
    """Extract child-indexed diffusion-edge metadata from a tree graph.

    Returns tensors suitable for registering on :class:`Tree`. The
    ``diff_parent_index`` and ``diff_geom_um`` buffers have length ``n_comp`` and
    are child-indexed: root entries have parent ``-1`` and zero geometry.
    Compact edge-list buffers are also returned for scatter/gather backends.
    Their edge axis is ordered by increasing child storage index, with the root
    omitted; :attr:`Tree.material_edge_index` exposes that ordering publicly for
    edge-located material data.
    """
    n_comp = len(graph.nodes)
    parent_index = [-1 for _ in range(n_comp)]
    diff_geom = [0.0 for _ in range(n_comp)]
    edge_parent, edge_child, edge_diff_geom = [], [], []

    for child in range(n_comp):
        preds = list(graph.predecessors(child))
        if not preds:
            continue
        if len(preds) > 1:
            raise ValueError(
                f"Node {child!r} has {len(preds)} parents; Tree diffusion requires "
                "a rooted tree morphology."
            )
        parent = int(preds[0])
        geom = _edge_diff_geom_um(graph, parent, child)
        parent_index[child] = parent
        diff_geom[child] = geom
        edge_parent.append(parent)
        edge_child.append(int(child))
        edge_diff_geom.append(geom)

    return {
        "diff_parent_index": torch.tensor(parent_index, dtype=torch.long),
        "diff_geom_um": torch.tensor(diff_geom, dtype=torch.float64).unsqueeze(0),
        "diff_edge_parent": torch.tensor(edge_parent, dtype=torch.long),
        "diff_edge_child": torch.tensor(edge_child, dtype=torch.long),
        "diff_edge_geom_um": torch.tensor(
            edge_diff_geom, dtype=torch.float64
        ).unsqueeze(0),
    }


def gather_membrane(graph):
    """Extract membrane parameters from a graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        Morphology graph whose nodes provide membrane attributes.

    Returns
    -------
    dict
        Mapping from attribute names to tensors shaped ``(1, n_comp)``.
    """
    rhoa, cm, area = [], [], []
    for i in range(len(graph.nodes)):
        attrs = graph.nodes[i]
        rhoa.append(attrs.get("Ra"))
        cm.append(attrs.get("cm"))
        area.append(attrs.get("area"))
    return {
        "rhoa": torch.tensor(rhoa, dtype=torch.float64).unsqueeze(0),
        "cm": torch.tensor(cm, dtype=torch.float64).unsqueeze(0),
        "area": torch.tensor(area, dtype=torch.float64).unsqueeze(0),
    }


class Tree(Population):
    """Base class for tree-like population models.
    All neuron morpholgies can be represented as trees of connected compartments.
    `Dendra` Axon classes specifically model unbranched axons, while `Tree` models
    can represent arbitrary tree-like morphologies with branching structures.

    Parameters
    ----------
    N : int
        Number of population instances.
    C : int
        Number of compartments per population.
    graph : networkx.DiGraph, optional
        Morphology graph describing tree structure.
    integrator : callable, optional
        Integrator factory used for simulation.
    **kwargs
        Additional parameters forwarded to :class:`Population`.
    """

    def __init__(
        self, N, C, graph=None, integrator=None, principal_axis=None, **kwargs
    ):
        if graph is not None:
            graph = _normalize_tree_graph(graph)
            if len(graph.nodes) != int(C):
                raise ValueError(
                    f"Tree graph has {len(graph.nodes)} compartments, but C={C}."
                )
        if integrator is None:
            integrator = dhs()
        super().__init__(N, C, integrator=integrator, **kwargs)
        self._graph = graph
        # A Tree is a compiled simulation object. Retain an independent graph
        # snapshot even for the documented low-level constructor; the public
        # NetworkX graph remains a mutable interoperability view only.
        self._compiled_graph = None if graph is None else graph.copy()
        self._compartment_graph = None
        names = []

        if graph is not None:
            for i in range(len(graph.nodes)):
                attrs = graph.nodes[i]
                name = attrs.get("name")
                names.append(name)
        self.names = names

        if principal_axis is not None:
            directions = (
                torch.as_tensor(
                    principal_axis, dtype=self.dtype(), device=self.device()
                )
                .reshape(1, 3)
                .expand(N, -1)
            ).clone()
        else:
            directions = (
                torch.tensor(
                    [[0.0, 0.0, 1.0]], dtype=self.dtype(), device=self.device()
                )
                .expand(N, -1)
                .clone()
            )

        self.register_buffer("directions", directions)
        self.register_buffer(
            "azimuthal_rotations",
            torch.tensor(0.0, dtype=self.dtype(), device=self.device())
            .expand(N)
            .clone(),
        )

        self.register_buffer("base_direction", self.directions.clone())
        self.register_buffer(
            "base_azimuthal_rotation", self.azimuthal_rotations.clone()
        )

        if graph is not None:
            diffusion_edges = gather_diffusion_edges(graph)
            for key, value in diffusion_edges.items():
                if value.dtype.is_floating_point:
                    if value.ndim == 2:
                        value = value.expand(N, -1).clone()
                    else:
                        value = value.clone()
                    value = value.to(device=self.device(), dtype=self.dtype())
                else:
                    value = value.clone().to(device=self.device())
                self.register_buffer(key, value)

        self[:, self.find_not("branchpoint")].label("internal_nodes")

    @property
    def graph(self):
        """networkx.DiGraph: Mutable morphology interoperability view.

        Simulation topology is the immutable compiled :attr:`compartment_graph`
        (and its registered topology buffers). Mutating this NetworkX view does
        not recompile the Tree; construct a new Tree to change morphology.
        """
        return self._graph

    @property
    def compartment_graph(self):
        """Canonical immutable morphology snapshot aligned to model storage.

        The legacy :attr:`graph` view remains available for NetworkX-based
        interoperability. ``compartment_graph`` records the validated geometry,
        topology, material domains, and provenance used at construction time.
        """
        return self._compartment_graph

    @property
    def area(self):
        """Exact compiled membrane area in square centimetres.

        A Tree is a compiled simulation object.  Its public NetworkX ``graph``
        remains mutable for interoperability, but changing that view must not
        change the numerical model piecemeal.  Read membrane area from the same
        immutable :class:`CompartmentGraph` snapshot that owns electrical
        topology and axial resistance.
        """
        graph = self.compartment_graph
        if graph is None:
            return super().area
        area = torch.as_tensor(
            graph.geometry.area_um2,
            device=self.device(),
            dtype=self.dtype(),
        ).reshape(1, -1)
        return (area * 1.0e-8).expand(self.np, -1)

    @property
    def edge_resistance_ohm(self):
        """Exact child-indexed compiled axial resistance in ohms.

        The root entry is zero; every other entry is the complete resistance
        between that compartment and its parent.  Values are shared across the
        population axis and retain the model's current dtype and device.
        """
        graph = self.compartment_graph
        if graph is None:
            raise AttributeError(
                "Tree has no compiled CompartmentGraph electrical geometry"
            )
        resistance = torch.as_tensor(
            graph.geometry.edge_resistance_ohm,
            device=self.device(),
            dtype=self.dtype(),
        ).reshape(1, -1)
        return resistance.expand(self.np, -1)

    @property
    def material_edge_index(self) -> torch.LongTensor:
        """Return the compact parent-child material edge index.

        Returns
        -------
        torch.LongTensor
            Tensor with shape ``(2, E)``. Row zero contains parent compartment
            indices and row one contains the corresponding child compartment
            indices. A connected Tree with ``C`` compartments has ``E = C - 1``
            edges.

        Notes
        -----
        Edges are oriented away from the morphology root and ordered by
        increasing child storage index, with the root omitted. This edge axis
        is the stable ordering for edge-located material geometry and
        diffusivity. Topology is shared by every population instance and batch
        replica, so this tensor has no population or batch axis. It is derived
        from registered topology buffers, follows model device moves, and does
        not add duplicate checkpoint state.
        """
        return torch.stack((self.diff_edge_parent, self.diff_edge_child), dim=0)

    @property
    def material_edges(self) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Return compact material edges as ``(parent, child)`` tensors.

        This is the unpacking-oriented form of :attr:`material_edge_index`::

            parent, child = tree.material_edges

        Both tensors have shape ``(E,)`` and use the same stable compact edge
        ordering documented by :attr:`material_edge_index`.
        """
        edge_index = self.material_edge_index
        return edge_index[0], edge_index[1]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Reject checkpoint topology that disagrees with this compiled Tree.

        ``graph`` remains a mutable NetworkX interoperability view, whereas the
        canonical ``CompartmentGraph`` and registered parent/edge buffers are
        the compiled morphology contract. Loading same-shaped buffers from a
        different Tree must not silently make chemical and electrical topology
        disagree.
        """
        if self._compartment_graph is not None:
            expected_parent = torch.as_tensor(
                self._compartment_graph.topology.parent_index, dtype=torch.long
            )
        elif hasattr(self, "diff_parent_index"):
            # Compatibility for specialized/legacy Tree factories without a
            # canonical CompartmentGraph snapshot: the target's registered
            # topology is still immutable during state restoration.
            expected_parent = self.diff_parent_index.detach().cpu().to(torch.long)
        else:
            expected_parent = None

        if expected_parent is not None:
            expected_child = torch.nonzero(
                expected_parent >= 0, as_tuple=False
            ).flatten()
            expected_edge_parent = expected_parent.index_select(0, expected_child)
            expected = {
                "diff_parent_index": expected_parent,
                "diff_edge_parent": expected_edge_parent,
                "diff_edge_child": expected_child,
            }
            for name, expected_value in expected.items():
                key = f"{prefix}{name}"
                incoming = state_dict.get(key)
                # Let the ordinary strict/non-strict loader report a missing
                # legacy key. When present, however, topology is immutable.
                if incoming is None:
                    continue
                if (
                    not torch.is_tensor(incoming)
                    or incoming.dtype != torch.long
                    or not torch.equal(incoming.detach().cpu(), expected_value)
                ):
                    raise RuntimeError(
                        "Cannot load a Tree checkpoint with a different or "
                        f"corrupt compiled material topology ({name!r})."
                    )

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def material_volume(self, domain="intracellular"):
        """Return the volume/mass buffer appropriate for a material domain.

        Parameters
        ----------
        domain : str, optional
            Domain name or alias. ``"i"``, ``"inside"``, ``"cytosol"``, and
            ``"intracellular"`` return ``volume_i``. ``"o"`` and
            ``"extracellular"`` return ``volume_o``. ``"total"`` returns
            ``volume``. ``"membrane"``/``"surface"`` return membrane area, which
            is useful for future surface-density material processes.
        """
        domain = str(domain or "intracellular").lower()
        if domain in {"i", "inside", "cytosol", "cytoplasm", "intracellular"}:
            return self.volume_i
        if domain in {"o", "outside", "extracellular"}:
            return self.volume_o
        if domain in {"total", "volume", "all"}:
            return self.volume
        if domain in {"membrane", "surface", "area"}:
            return self.area
        raise ValueError(f"Unsupported material domain {domain!r} for Tree.")

    @classmethod
    def from_graph(cls, graph, N=1, integrator=None, **kwargs):
        """Instantiate a tree population from a morphology graph.

        Parameters
        ----------
        graph : networkx.DiGraph
            Connected directed tree describing compartment connectivity. Node
            labels that are not already ``0..n-1`` are deterministically
            relabelled in graph insertion order on an internal copy.
        N : int, optional
            Number of population instances. Defaults to ``1``.
        integrator : callable, optional
            Integrator factory. Defaults to :func:`dendra.models.integrators.dhs`.
        **kwargs
            Additional membrane parameters forwarded to :class:`Tree`.

        Returns
        -------
        Tree
            Configured tree population.

        Raises
        ------
        ValueError
            If the graph is empty, cyclic, disconnected, contains a self-loop,
            or gives a node more than one parent.
        """
        graph = _normalize_tree_graph(graph)
        compartment_graph = CompartmentGraph.from_networkx(graph)
        C = len(graph.nodes)
        data = gather_morphology(graph)
        diffusion_edges = gather_diffusion_edges(graph)
        membrane = gather_membrane(graph)
        membrane.update(kwargs)
        tree = cls(N, C, graph, integrator, **membrane)
        for key, value in data.items():
            tree.register_buffer(
                key,
                value.expand(N, -1)
                .clone()
                .to(device=tree.device(), dtype=tree.dtype()),
            )
        for key, value in diffusion_edges.items():
            if value.dtype.is_floating_point:
                if value.ndim == 2:
                    value = (
                        value.expand(N, -1)
                        .clone()
                        .to(device=tree.device(), dtype=tree.dtype())
                    )
                else:
                    value = value.clone().to(device=tree.device(), dtype=tree.dtype())
            else:
                value = value.clone().to(device=tree.device())
            if key in tree._buffers:
                tree._buffers[key] = value
            else:
                tree.register_buffer(key, value)
        tree.slice("soma").label("soma")
        tree.slice("axon").label("axon")
        tree.slice("dend").label("dend")
        tree.slice("apic").label("apic")
        tree._compartment_graph = compartment_graph
        _register_canonical_internal_nodes(tree, compartment_graph)
        _register_compartment_graph_labels(tree, compartment_graph)
        return tree

    @classmethod
    def from_compartment_graph(cls, graph, N=1, integrator=None, **kwargs):
        """Construct a Tree from Dendra's canonical compartment graph.

        Parameters
        ----------
        graph : dendra.models.morphology.CompartmentGraph
            Immutable scalar compartment resistor tree.
        N : int, optional
            Number of copies of the morphology.
        integrator : callable, optional
            Scalar tree integrator factory.
        **kwargs
            Membrane parameter overrides forwarded to :meth:`from_graph`.
        """
        if cls.from_graph.__func__ is not Tree.from_graph.__func__:
            raise NotImplementedError(
                f"{cls.__name__}.from_compartment_graph requires an explicit "
                "adapter because this subclass defines custom from_graph semantics."
            )
        if not isinstance(graph, CompartmentGraph):
            raise TypeError("graph must be a CompartmentGraph.")
        tree = cls.from_graph(graph.to_networkx(), N=N, integrator=integrator, **kwargs)
        tree._compartment_graph = graph
        _register_compartment_graph_labels(tree, graph)
        return tree

    @classmethod
    def from_morphology(cls, morphology, N=1, integrator=None, **kwargs):
        """Discretize a native Section morphology and construct a Tree.

        Native and NEURON-authored morphologies converge on the same scalar
        compartment resistor-graph contract. Compilation is a snapshot:
        subsequently adding or updating Sections on the source builder cannot
        mutate the constructed model. Call ``from_morphology`` again to build
        a model from the revised declaration.

        Parameters
        ----------
        morphology : dendra.models.morphology.Morphology
            Connected native Section tree with explicit ``nseg`` values.
        N : int, optional
            Number of population instances.
        integrator : callable, optional
            Scalar tree integrator factory.
        **kwargs
            Membrane parameter overrides forwarded to :meth:`from_graph`.
        """
        if not isinstance(morphology, Morphology):
            raise TypeError("morphology must be a Morphology.")
        graph = morphology.compile()
        tree = cls.from_compartment_graph(graph, N=N, integrator=integrator, **kwargs)

        ordered_labels = {}
        for section in morphology.sections:
            section_nodes = sorted(
                (
                    node
                    for node, section_name in enumerate(graph.metadata.section_name)
                    if section_name == section.name
                ),
                key=lambda node: graph.metadata.segment_index[node],
            )
            for label in section.labels:
                ordered_labels.setdefault(label, []).extend(section_nodes)
        _register_compartment_graph_labels(tree, graph, ordered=ordered_labels)
        return tree

    @classmethod
    def from_NEURON(
        cls,
        root_sec=None,
        N=1,
        integrator=None,
        principal_axis=None,
        exclude=None,
        **kwargs,
    ):
        """Construct a tree population from a NEURON root section.

        Parameters
        ----------
        root_sec : neuron.h.Section, optional
            Root section of a NEURON morphology.
        N : int, optional
            Number of population instances. Defaults to ``1``.
        integrator : callable, optional
            Integrator factory for the population.
        **kwargs
            Additional keyword arguments forwarded to :meth:`from_graph`.

        Returns
        -------
        Tree
            Configured tree population.
        """
        from dendra.models.io import neuron_to_dendra_graph

        graph, _ = neuron_to_dendra_graph(root_sec, exclude=exclude)
        cell = cls.from_graph(
            graph, N, integrator, principal_axis=principal_axis, **kwargs
        )
        return cell

    @classmethod
    def from_swc(
        cls,
        file_path,
        d_lambda=0.1,
        freq=100.0,
        N=1,
        integrator=None,
        principal_axis=None,
        **kwargs,
    ):
        """Construct a tree population from an SWC file.

        Parameters
        ----------
        file_path : str
            Path to the SWC morphology file.
        d_lambda : float, optional
            Dimensionless spatial-discretisation factor for tree reconstruction.
        freq : float, optional
            Frequency in hertz used by NEURON's d-lambda morphology
            discretisation. Pass a raw hertz value such as ``freq=100.0``.
            This morphology-only argument is intentionally an exception to
            Dendra's usual kHz (1/ms) frequency convention, so do not multiply
            it by :data:`dendra.units.Hz`.
        N : int, optional
            Number of population instances. Defaults to ``1``.
        integrator : callable, optional
            Integrator factory for the population.
        **kwargs
            Additional keyword arguments forwarded to :meth:`from_graph`.

        Returns
        -------
        Tree
            Configured tree population.
        """
        from dendra.models.io import read_swc

        graph, _ = read_swc(file_path, d_lambda=d_lambda, freq=freq, **kwargs)
        cell = cls.from_graph(
            graph, N, integrator, principal_axis=principal_axis, **kwargs
        )
        return cell

    @classmethod
    def from_neurolucida(
        cls,
        file_path,
        d_lambda=0.1,
        freq=100.0,
        N=1,
        integrator=None,
        principal_axis=None,
        **kwargs,
    ):
        """Construct a tree population from a Neurolucida file.

        Parameters
        ----------
        file_path : str
            Path to the Neurolucida morphology file.
        d_lambda : float, optional
            Dimensionless spatial-discretisation factor for tree reconstruction.
        freq : float, optional
            Frequency in hertz used by NEURON's d-lambda morphology
            discretisation. Pass a raw hertz value such as ``freq=100.0``.
            This morphology-only argument is intentionally an exception to
            Dendra's usual kHz (1/ms) frequency convention, so do not multiply
            it by :data:`dendra.units.Hz`.
        N : int, optional
            Number of population instances. Defaults to ``1``.
        integrator : callable, optional
            Integrator factory for the population.
        **kwargs
            Additional keyword arguments forwarded to :meth:`from_graph`.

        Returns
        -------
        Tree
            Configured tree population.
        """
        from dendra.models.io import read_neurolucida

        graph, _ = read_neurolucida(file_path, d_lambda=d_lambda, freq=freq, **kwargs)
        cell = cls.from_graph(
            graph, N, integrator, principal_axis=principal_axis, **kwargs
        )
        return cell

    from_asc = from_neurolucida

    def recentre(self, x=0.0, y=0.0, z=0.0, origin=None):
        """Recentre the morphology so the soma matches ``origin``.

        Parameters
        ----------
        x : float or torch.Tensor, optional
            Target x-coordinate for the soma.
        y : float or torch.Tensor, optional
            Target y-coordinate for the soma.
        z : float or torch.Tensor, optional
            Target z-coordinate for the soma.
        origin : int or None, optional
            Compartment index of the soma. Defaults to the middle soma node.

        Returns
        -------
        Tree
            Modified instance for chaining.
        """
        x = torch.as_tensor(x, dtype=self.x.dtype, device=self.x.device)
        y = torch.as_tensor(y, dtype=self.y.dtype, device=self.y.device)
        z = torch.as_tensor(z, dtype=self.z.dtype, device=self.z.device)

        ndim_required = self.x.ndim - 1

        assert x.ndim == ndim_required or x.ndim == 0, (
            f"Expected x.ndim to be {ndim_required} or 0, but got {x.ndim}"
        )
        assert y.ndim == ndim_required or y.ndim == 0, (
            f"Expected y.ndim to be {ndim_required} or 0, but got {y.ndim}"
        )
        assert z.ndim == ndim_required or z.ndim == 0, (
            f"Expected z.ndim to be {ndim_required} or 0, but got {z.ndim}"
        )

        if x.ndim != 0:
            x = x.unsqueeze(-1)
        if y.ndim != 0:
            y = y.unsqueeze(-1)
        if z.ndim != 0:
            z = z.unsqueeze(-1)

        if origin is None:
            origin = self.find("soma", as_list=True)
            origin = origin[int(len(origin) / 2)]

        current_centre_x = self.x[..., origin].unsqueeze(-1)
        current_centre_y = self.y[..., origin].unsqueeze(-1)
        current_centre_z = self.z[..., origin].unsqueeze(-1)

        offsets = [x - current_centre_x, y - current_centre_y, z - current_centre_z]

        self.x += offsets[0]
        self.y += offsets[1]
        self.z += offsets[2]

        return self

    def shift(self, dx=0.0, dy=0.0, dz=0.0):
        """Translate the morphology by the specified offsets.

        Parameters
        ----------
        dx : float or torch.Tensor, optional
            Offset along the x-axis.
        dy : float or torch.Tensor, optional
            Offset along the y-axis.
        dz : float or torch.Tensor, optional
            Offset along the z-axis.

        Returns
        -------
        Tree
            Modified instance for chaining.
        """

        dx = torch.as_tensor(dx, dtype=self.x.dtype, device=self.x.device)
        dy = torch.as_tensor(dy, dtype=self.y.dtype, device=self.y.device)
        dz = torch.as_tensor(dz, dtype=self.z.dtype, device=self.z.device)

        ndim_required = self.x.ndim - 1

        assert dx.ndim == ndim_required or dx.ndim == 0, (
            f"Expected dx.ndim to be {ndim_required} or 0, but got {dx.ndim}"
        )
        assert dy.ndim == ndim_required or dy.ndim == 0, (
            f"Expected dy.ndim to be {ndim_required} or 0, but got {dy.ndim}"
        )
        assert dz.ndim == ndim_required or dz.ndim == 0, (
            f"Expected dz.ndim to be {ndim_required} or 0, but got {dz.ndim}"
        )

        if dx.ndim != 0:
            dx = dx.unsqueeze(-1)
        if dy.ndim != 0:
            dy = dy.unsqueeze(-1)
        if dz.ndim != 0:
            dz = dz.unsqueeze(-1)

        self.x += dx
        self.y += dy
        self.z += dz

        return self

    def move_to(self, x=0.0, y=0.0, z=0.0, origin=None):
        """Move the morphology so the soma lies at ``(x, y, z)``.

        Returns
        -------
        Tree
            Modified instance for chaining.
        """
        return self.recentre(x, y, z, origin)

    def _get_points_as_tensor(self) -> torch.Tensor:
        """Stack coordinates as ``(..., neuron, compartment, xyz)``."""
        return torch.stack([self.x, self.y, self.z], dim=-1)

    def _update_points_from_tensor(self, points: torch.Tensor):
        """Copy a trailing xyz coordinate dimension back into model buffers."""
        self.x.copy_(points[..., 0])
        self.y.copy_(points[..., 1])
        self.z.copy_(points[..., 2])

    def _apply_rotation(self, rotation_matrices: torch.Tensor, origin_idx: int):
        """
        Applies a batch of rotation matrices to the cell compartments.

        Args:
            rotation_matrices (torch.Tensor): A (B, 3, 3) tensor of rotation matrices.
            origin_idx (int): The index of the compartment to use as the rotation origin.
        """
        points = self._get_points_as_tensor()

        # Select the compartment axis while preserving any leading parameter
        # batches and the physical-neuron axis.
        origins = points[..., origin_idx, :].clone().unsqueeze(-2)

        # 2. Translate points so the origin is at (0,0,0)
        points_centered = points - origins

        # Apply one rotation per physical neuron, broadcasting it over any
        # leading parameter-batch dimensions and all compartments.
        rotated_points_centered = points_centered @ rotation_matrices.transpose(
            -1, -2
        ).to(points.dtype)

        # 4. Translate points back
        rotated_points = rotated_points_centered + origins

        # 5. Update the internal buffers
        self._update_points_from_tensor(rotated_points)

    def rotate_into_direction(
        self, target_directions: torch.Tensor, origin: int = None
    ):
        """
        Rotates cells to align their current directions with target directions.

        Args:
            target_directions (torch.Tensor): A (B, 3) or (1, 3) tensor of target directions.
            origin_idx (int): The index of the compartment to use as the rotation origin.
        """
        target_directions = torch.as_tensor(
            target_directions,
            dtype=self.directions.dtype,
            device=self.directions.device,
        ).reshape(-1, 3)

        if origin is None:
            origin = self.find("soma", as_list=True)
            origin = origin[int(len(origin) / 2)]

        device = self.directions.device

        if target_directions.shape[0] == 1:
            target_directions = target_directions.repeat(self.np, 1)
        elif target_directions.shape[0] != self.np:
            raise ValueError(
                "target_directions must contain either one direction or one "
                f"direction per cell ({self.np}); got {target_directions.shape[0]}."
            )
        target_directions = target_directions.to(device)

        if torch.any(torch.linalg.vector_norm(self.directions, dim=1) == 0):
            raise ValueError("Current cell directions must be non-zero vectors.")
        if torch.any(torch.linalg.vector_norm(target_directions, dim=1) == 0):
            raise ValueError("Target cell directions must be non-zero vectors.")

        a = F.normalize(self.directions, p=2, dim=1)
        b = F.normalize(target_directions, p=2, dim=1)

        # --- Use Rodrigue's formula to get the rotation matrix R ---
        # c is the cosine of the angle (dot product), shape (B,)
        c = torch.sum(a * b, dim=1).clamp(-1.0, 1.0)

        # The cross product contains both the rotation axis and a numerically
        # stable sin(theta). Only truly degenerate axes need special handling;
        # a fixed cosine threshold incorrectly turns small requested rotations
        # into identity transforms while still updating direction metadata.
        v = torch.cross(a, b, dim=1)
        s = torch.linalg.vector_norm(v, dim=1)
        tolerance = 10 * torch.finfo(self.directions.dtype).eps
        is_degenerate = s <= tolerance
        is_identity = is_degenerate & (c >= 0)
        is_anti_parallel = is_degenerate & (c < 0)

        # Handle the anti-parallel case where the cross product is near zero
        if torch.any(is_anti_parallel):
            # Find an arbitrary perpendicular axis for the 180-degree rotation
            temp_vec = (
                torch.tensor(
                    [1.0, 0.0, 0.0], device=device, dtype=self.directions.dtype
                )
                .expand(self.np, -1)
                .clone()
            )
            parallel_to_temp = torch.all(
                torch.isclose(a, temp_vec) | torch.isclose(a, -temp_vec), dim=1
            )
            temp_vec[parallel_to_temp] = torch.tensor(
                [0.0, 1.0, 0.0], device=device, dtype=self.directions.dtype
            )

            v[is_anti_parallel] = F.normalize(
                torch.cross(a[is_anti_parallel], temp_vec[is_anti_parallel], dim=1),
                dim=1,
            )

        # Rodrigues' formula below expects a unit rotation axis. Degenerate
        # identity rows deliberately retain the zero axis; their matrices are
        # replaced explicitly below.
        safe_s = s.clamp_min(torch.finfo(self.directions.dtype).tiny)
        v = torch.where(is_degenerate.unsqueeze(1), v, v / safe_s.unsqueeze(1))

        # Skew-symmetric cross-product matrix K
        K = torch.zeros(self.np, 3, 3, device=device, dtype=self.directions.dtype)
        K[:, 0, 1] = -v[:, 2]
        K[:, 0, 2] = v[:, 1]
        K[:, 1, 0] = v[:, 2]
        K[:, 1, 2] = -v[:, 0]
        K[:, 2, 0] = -v[:, 1]
        K[:, 2, 1] = v[:, 0]

        # --- Now, reshape for the main formula ---
        # Using new names for clarity
        s_mat = s.view(self.np, 1, 1)
        c_mat = c.view(self.np, 1, 1)

        I = torch.eye(3, device=device, dtype=self.directions.dtype).expand(
            self.np, -1, -1
        )  # noqa: E741
        R = I + s_mat * K + (1 - c_mat) * (K @ K)

        # --- Apply special cases using the (B,) shaped masks ---
        # This is now correct because `is_identity` has shape (B,)
        R[is_identity] = torch.eye(3, device=device, dtype=self.directions.dtype)

        # This was already correct, but the logic is now more robust
        if torch.any(is_anti_parallel):
            v_ap = v[is_anti_parallel]
            # Formula for 180-degree rotation matrix around axis v
            R_ap = 2 * torch.einsum("bi,bj->bij", v_ap, v_ap) - torch.eye(
                3, device=device, dtype=self.directions.dtype
            )
            R[is_anti_parallel] = R_ap

        self._apply_rotation(R, origin)

        # Update the cell's direction vector
        # We use b, the normalized target, for consistency
        self.directions.copy_(b)
        return self

    def rotate_azimuthal(
        self, azimuthal_angle: float | torch.Tensor, origin: int = None
    ):
        """
        Rotates cells around their current direction vector by a given angle.

        Args:
            azimuthal_angle (float or torch.Tensor): Angle in degrees. Can be a single
                                                     float or a (B,) tensor for individual angles.
            origin_idx (int): The index of the compartment to use as the rotation origin.
        """
        if origin is None:
            origin = self.find("soma", as_list=True)
            origin = origin[int(len(origin) / 2)]

        device = self.directions.device

        # Axis of rotation is the cell's own direction
        v = F.normalize(self.directions, p=2, dim=1)

        # Convert angle to radians and ensure it's a (B,) tensor
        theta = torch.as_tensor(
            azimuthal_angle, device=device, dtype=self.directions.dtype
        )
        if theta.numel() == 1:
            theta = theta.reshape(()).expand(self.np).clone()
        elif theta.numel() == self.np:
            theta = theta.reshape(self.np)
        else:
            raise ValueError(
                "azimuthal_angle must be scalar or contain one angle per cell "
                f"({self.np}); got {theta.numel()} values."
            )
        theta_rad = torch.deg2rad(theta)

        c = torch.cos(theta_rad)
        s = torch.sin(theta_rad)

        # Skew-symmetric cross-product matrix K
        K = torch.zeros(self.np, 3, 3, device=device, dtype=self.directions.dtype)
        K[:, 0, 1] = -v[:, 2]
        K[:, 0, 2] = v[:, 1]
        K[:, 1, 0] = v[:, 2]
        K[:, 1, 2] = -v[:, 0]
        K[:, 2, 0] = -v[:, 1]
        K[:, 2, 1] = v[:, 0]

        s = s.view(self.np, 1, 1)
        c = c.view(self.np, 1, 1)

        I = torch.eye(3, device=device, dtype=self.directions.dtype).expand(
            self.np, -1, -1
        )  # noqa: E741
        R = I + s * K + (1 - c) * (K @ K)

        self._apply_rotation(R, origin)
        # Note: self.directions does NOT change in an azimuthal rotation
        self.azimuthal_rotations.add_(theta)
        self.azimuthal_rotations.copy_(
            (self.azimuthal_rotations + 180.0) % 360.0 - 180.0
        )
        return self

    def reset_rotations(self, origin=None):
        if origin is None:
            origin = self.find("soma", as_list=True)
            origin = origin[int(len(origin) / 2)]
        x_c = self.x[..., origin]
        y_c = self.y[..., origin]
        z_c = self.z[..., origin]

        # Reset directions to the base direction
        self.directions.copy_(self.base_direction.expand(self.np, -1))
        # Reset azimuthal rotations to the base azimuthal rotation
        self.azimuthal_rotations.copy_(self.base_azimuthal_rotation.expand(self.np))

        morph = gather_morphology(self.graph)
        base_x = morph["x"].to(dtype=self.x.dtype, device=self.x.device)
        base_y = morph["y"].to(dtype=self.y.dtype, device=self.y.device)
        base_z = morph["z"].to(dtype=self.z.dtype, device=self.z.device)
        self.x.copy_(
            (base_x - base_x[..., origin].unsqueeze(-1)).expand_as(self.x)
            + x_c.unsqueeze(-1)
        )
        self.y.copy_(
            (base_y - base_y[..., origin].unsqueeze(-1)).expand_as(self.y)
            + y_c.unsqueeze(-1)
        )
        self.z.copy_(
            (base_z - base_z[..., origin].unsqueeze(-1)).expand_as(self.z)
            + z_c.unsqueeze(-1)
        )

        return self

    def reset_directions(self, origin: int = None):
        self.rotate_into_direction(self.base_direction, origin)
        return self

    def reset_azimuthal_rotations(self, origin: int = None):
        angles_to_undo = -self.azimuthal_rotations.clone()
        self.rotate_azimuthal(angles_to_undo, origin)
        self.azimuthal_rotations.copy_(self.base_azimuthal_rotation.expand(self.np))
        return self
