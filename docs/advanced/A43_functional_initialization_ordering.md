---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-initialization-ordering)=
# Initialize shared Network parameters in order

`dn.compile(network, initialization="fresh", ...)` initializes Populations in
their authored Network order. Each Population prepares its numerical operands
at its own initialization boundary. A later Population can update a shared raw
parameter without replacing the earlier Population's prepared values.

The complete
{download}`functional_initialization_ordering.py <../../examples/functional_initialization_ordering.py>`
example compares a two-Population rollout, shared-parameter gradients, recording,
and explicit continuation against native execution. Run it with
`python examples/functional_initialization_ordering.py`.

## Share a source and register a transform

An initialization transform returns new tensors for its declared writes:

```{literalinclude} ../../examples/functional_initialization_ordering.py
:language: python
:start-after: "# region transform"
:end-before: "# endregion transform"
```

This Network shares one capacitance source. The second Population transforms
that source during its own initialization:

```{literalinclude} ../../examples/functional_initialization_ordering.py
:language: python
:start-after: "# region sharing"
:end-before: "# endregion sharing"
:dedent: 4
```

The first Population prepares its capacitance and solver coefficients before
the second Population's write. The second prepares its own values after the
write. Both parameter aliases refer to the final shared raw tensor, while the
Populations retain their individual prepared operands. Registered post-initialize
parameter transforms rematerialize their own Population before its solver is
configured, following the same native lifecycle.

Sharing the same registered tensor as a parameter and another Population's
configuration buffer also follows this order: readers initialized before the
write keep their prepared values; readers initialized afterward see the new
value. Computed PyTorch parametrization properties retain their getter semantics.
See {doc}`A38_functional_torch_parametrizations` for raw originals and transforms.
Population RAND and NOISE sampling precedes pre-initialize parameter writes;
inserted owners sample afterward. See
{doc}`A45_functional_stochastic_initialization_ordering` for shared stochastic
distribution sources on both sides of a transform.

## Differentiate and continue a rollout

Select the shared conductance using either qualified alias:

```{literalinclude} ../../examples/functional_initialization_ordering.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 8
```

Fresh initialization uses current raw parameters on each restart. Its returned
constant frame contains the per-Population prepared values, including their
initialization computation graphs. Shared raw aliases remain shared, and
gradients accumulate across their uses. Differentiation through a later
transform's explicit inputs affects only the values prepared after that write.
With the pure `functional.initialize(...)` API, differentiate with respect to
its input parameter tensors or transform inputs. The returned parameter frame
can contain transformed or cache-restored tensors; earlier preparation can
depend on the original inputs before those replacements. The compiled runner's
selected trainable leaves are these original inputs.

Carry the returned parameters and constants with the runtime state when
continuing:

```{literalinclude} ../../examples/functional_initialization_ordering.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 8
```

These packets preserve initialization decisions across scan chunks,
checkpoint recomputation, and inference while loops. After an optimizer update,
start another fresh run to prepare operands from the updated parameters.

Snapshot execution preserves the captured native preparation baseline while
allowing explicit raw parameter changes to affect subsequent preparation.
Each extracted packet carries its own baseline, including when
`functional.extract(other_network)` reads a compatible Network. Extracting
another packet does not change earlier packets or the plan's default baseline.
Fresh initialization rebuilds from current raw inputs and its initialization
packet.

`commit_state_(network, state)` copies runtime carry into a compatible native
Network. Prepare that destination with matching parameters and initialization
before committing; this method does not apply a functional parameter or constant
packet. A functional checkpoint includes all three packets for continuation.

## Mix cached and fresh Populations

Without a Network-global cache, a Population's steady-state cache restores at
that Population's position in the same ordered transaction. A later cached
parameter write leaves an earlier fresh Population's prepared values intact.
Shared effective-buffer storage follows native copy semantics: a later copy can
change a live buffer while an earlier solver retains its previously prepared
coefficients. Cache-restored runtime values keep their native detached gradient
boundaries; fresh materializations retain their gradients.

A Network-global cache restores all Population state before solver setup, as in
native execution. See {doc}`A29_functional_network_cache_initialization` for
cache replacement and connection restoration. Initialization does not mutate
the source Network or incoming packets. Structural changes to transforms,
sharing, or cache ownership require lowering the Network again.

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_initialization_ordering.py
:tags: [remove-input]
```
