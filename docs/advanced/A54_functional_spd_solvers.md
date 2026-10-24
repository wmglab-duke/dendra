---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-spd-solvers)=
# Symmetric positive-definite solvers

`dn.func` and `dn.compile(...)` support the scalar and fixed 3×3 block SPD
solvers selected by `dn.bwd_euler_ub(method="spd")` and
`dn.bwd_euler_bt(method="spd")`. Compatible Populations can be lowered directly
or included in a Network with other supported integrators. Implicit Material
diffusion also supports `solver="spd"` on CPU.

CPU execution requires a `dendra-solvers` build that provides the pure SPD
facades. Native and functional execution use the same Cholesky factorization
and cached-factor solve kernels. Functional calls allocate private disposable
workspaces, leaving caller-owned coefficient and RHS tensors unchanged.

The complete
{download}`functional_spd_solvers.py <../../examples/functional_spd_solvers.py>`
example builds a connected scalar/block Network, compares native execution,
checks shared gradients, and continues a checkpointed rollout. Run it from the
repository root with `python examples/functional_spd_solvers.py`.

## Choose the solver when constructing a Population

The scalar solver operates on a tridiagonal system. The block solver operates
on an `ExtCellAxon` with two extracellular layers, giving three voltage unknowns
per compartment:

```{literalinclude} ../../examples/functional_spd_solvers.py
:language: python
:start-after: "# region models"
:end-before: "# endregion models"
:dedent: 8
```

The assembled matrix must satisfy the solver's symmetry and positive-definite
requirements. On CPU, scalar lower and upper bands must match exactly. Block
lower and upper bands must match, and each 3×3 main block must be symmetric.
Heterogeneous cable capacitance can produce nonsymmetric scalar bands; such a
system requires a solver that supports nonsymmetric matrices.

## Compile, train, and continue

Population-qualified selectors identify trainable parameters. The example
shares one leak parameter between both Populations:

```{literalinclude} ../../examples/functional_spd_solvers.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

`inference=False` uses differentiable scan execution. The same SPD preparation
and solve adapters serve inference through scan or `execution="while"`.
Shared raw parameters retain their identity, and current, geometry, stimulation,
and connection gradients pass through the solve:

```{literalinclude} ../../examples/functional_spd_solvers.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

Calling the runner without a state restarts from the selected initialization.
Supply its complete returned state and effective parameter/constant packets to
continue:

```{literalinclude} ../../examples/functional_spd_solvers.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

`initialization="fresh"`, Population-owned steady-state caches, Network caches,
and ordinary native checkpoint restoration keep their standard ordering and
parameter lookup semantics. Cholesky factors are local numerical workspaces;
they add no persistent simulation state or checkpoint fields. Committing the
functional state to a compatible native model allows ordinary checkpoint saving
and native continuation, as described in {doc}`A21_functional_network_compile`.

## Derivatives and numerical repair

SPD facades support first-order reverse and forward AD, `vjp`, `jvp`,
`jacrev`, `jacfwd`, and direct or nested `vmap`, including empty mapped lanes.
Cached SPD derivatives are first-order only.

Scalar and block off-diagonal bands have separate ambient derivatives, evaluated
on the symmetry-constrained matrix. When both bands share one parameter,
autograd sums their contributions. Block main gradients project onto symmetric
matrices; forward AD uses the symmetric part of the main-block tangent.

CPU inference can repair small or failed Cholesky pivots. A repaired solve
operates on the modified matrix and can differ from the original system. Repair
is disabled when coefficient gradients or active coefficient forward-AD
tangents are required; the solve raises if unrepaired factorization fails.
The scalar policy checks its lower and main bands, preserving the native
exception for the redundant upper band. Share the lower and upper tensors
when training a symmetry-preserving off-diagonal parameter. The block policy
checks all three coefficient tensors.
RHS-only derivatives permit repair. For an RHS-only JVP, keep coefficients
constant and differentiate a function whose argument is only the RHS.

Scalar CUDA requests retain the native Thomas fallback. Block CUDA uses the
native SPD kernel and its diagonal-repair policy. Its derivatives treat the
repair shift as fixed; when a repair is triggered they exclude the shift's
dependence on the coefficients. Use well-conditioned SPD systems when exact
coefficient sensitivities are required. MPS functional execution is currently
unsupported.
