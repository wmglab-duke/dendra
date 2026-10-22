---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-network-regions)=
# Record named regions in a Network

Use `dn.func.Recorder(..., region=...)` to record a physical Slice from one
Population in a functional Network. Keep the qualified state names that identify
the owning Population:

```python
callbacks = {
    "soma": dn.func.Recorder(("left.v", "t"), region=network.left.soma),
}
```

The callback name `"soma"` names the result stream. The state name `"left.v"`
selects a field, and `region` selects its physical coordinates. Scan training
and while inference use the same selector and return the same shape.

The complete
{download}`functional_network_regions.py <../../examples/functional_network_regions.py>`
example runs both execution modes for an ordinary Network and an actual
`Network.concat()` result. It checks named regions, membrane current, repeated
indices, batch selection, empty selections, gradients, and continuation. Run it
from the repository root with `python examples/functional_network_regions.py`.

## Label physical regions before concatenation

Define labels using the ordinary Population Slice API. Apply concatenation and
batching before initialization and functional lowering:

```{literalinclude} ../../examples/functional_network_regions.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
:dedent: 4
```

`Network.concat()` returns a new Network. With its default grouping, the source
Populations become named components of `network.all_populations`. Their labels
are available through that packed Population:

```python
network = network.concat()
network.batch(2)
network.initialize(dt)
callbacks = {
    "exc": dn.func.Recorder(
        ("all_populations.v",), region=network.all_populations.exc
    ),
    "exc_soma": dn.func.Recorder(
        ("all_populations.v", "t"), region=network.all_populations.exc.soma
    ),
}
```

The field prefix is the name of the Population in the transformed Network,
`all_populations`. Component names such as `exc` belong in the region path.
Packed voltage state remains a flat Tensor; it does not acquire a nested `exc`
state mapping. A component may contain several cells. Label a particular cell
before concatenation, such as `exc[1, :].label("cell_1")`, to select it through
`network.all_populations.exc.cell_1`.

Create each region from the exact source Network passed to `dn.compile` or
`make_functional_network`. An equivalent Slice from another Network, or a Slice
retained from before concatenation, does not identify the transformed owner.
Selection indices are copied when callbacks bind; edit the callback configuration
and bind again to change a selection.

## Select batches and preserve indexing shape

```{literalinclude} ../../examples/functional_network_regions.py
:language: python
:start-after: "# region regions"
:end-before: "# endregion regions"
:dedent: 4
```

Each recorded physical field has shape `(samples, *region.shape)`. A Slice keeps
PyTorch's physical indexing order, repeated indices, selected batch lanes,
inserted dimensions, and empty dimensions. Repeated coordinates remain repeated
in the result. An empty region produces an empty physical field with the normal
number of time samples.

For `vc`, selection acts on physical cell and compartment axes and retains its
three trailing circuit-potential channels. Its shape is
`(samples, *region.shape, 3)`. Concatenation flattens each component's original
cell and compartment axes; indexing a packed component does not reconstruct
those axes.

## Compile, record, and continue

```{literalinclude} ../../examples/functional_network_regions.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

Results keep their qualified field names. For example,
`aux["callbacks"]["soma"]["left.v"]` contains the selected voltages and
`aux["callbacks"]["soma"]["t"]` contains the Network clock. A scalar clock is
recorded once per frame and is not indexed by the region. Enable membrane-current
output in the Population integrator before recording `i_membrane`.

A fresh recording includes the initial frame and every accepted timestep.
Continue with both explicit simulation state and callback carry to avoid
duplicating the boundary frame:

```{literalinclude} ../../examples/functional_network_regions.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

With `execution="scan", inference=False`, selected traces retain gradients
through selection and continuation. Repeated indices accumulate their gradient
contributions at the repeated physical coordinates. While execution suppresses
gradients and retains the same recording and continuation semantics.

```{literalinclude} ../../examples/functional_network_regions.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

## Supported selections

Region selection supports physical `v`, `vc`, and `i_membrane` fields belonging
to the region's Population. Fully qualified paths such as
`populations.left.integrator.v` are also accepted. Global `t` and the selected
Population's clock, such as `left.t` or `populations.left.clock.t`, remain
scalar selections. A clock or physical field qualified with another Population
is rejected: each regional Recorder belongs to one physical owner.

Mechanism-local states are recordable without `region`. Region selection for
those states is currently unsupported because their support can differ from
physical Population coordinates. A region must select the Population itself,
rather than a mechanism submodule.

`region` cannot be combined with `node_indices`, `max_only`, or `partition`.
Those arguments retain their ordinary final-axis selection semantics; for
example, `node_indices` on `vc` selects circuit channels rather than physical
compartments. Use separate Recorder streams when different selections are
needed. Other Network callback contracts are described in
{doc}`A20_functional_network_rollout` and {doc}`A25_functional_network_while`.

The complete CPU example executes here:

```{code-cell} python
:load: ../../examples/functional_network_regions.py
:tags: [remove-input]
```
