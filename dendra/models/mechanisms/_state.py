import inspect
import textwrap
from types import MethodType

import torch

from dendra.models.parametric import Parameterized
from dendra.models.rng import RNGModule

from ._bufferimplicit import build_bufferimplicit
from ._cnexp import build_cnexp
from ._derivimplicit import build_derivimplicit
from ._euler_heun import build_euler_heun
from ._euler_maruyama import build_euler_maruyama
from ._kinetic import kinetic_to_derivatives
from ._linearimplicit import build_linearimplicit
from ._mechanism import classproperty
from ._rosenbrock import build_rosenbrock1

# Integration-method registry -------------------------------------------------
#
# State subclasses can declare their solver with State.METHOD(...).  The registry
# keeps method dispatch out of State.__init__ and makes it straightforward to add
# new generated solvers such as linearimplicit/sparse, rosenbrock1, or
# bufferimplicit without editing the State class again.
_INTEGRATION_BUILDERS = {}
_INTEGRATION_ALIASES = {}


def _canonical_method_name(method):
    if method is None:
        method = "cnexp"
    if not isinstance(method, str):
        raise TypeError(
            f"State integration method must be a string; got {type(method).__name__}."
        )
    method = method.strip().lower().replace("-", "_")
    return _INTEGRATION_ALIASES.get(method, method)


def register_integration_method(name, builder, *, aliases=()):
    """Register a generated integration-function builder.

    Parameters
    ----------
    name : str
        Canonical method name, e.g. ``"cnexp"`` or ``"derivimplicit"``.
    builder : callable
        Function with signature ``builder(states, assigned, derivative,
        eliminate=None, **method_kwargs)`` returning a generated ``solve``
        function.
    aliases : iterable[str], optional
        Additional names accepted by :meth:`State.METHOD`.
    """
    canonical = str(name).strip().lower().replace("-", "_")
    if not canonical:
        raise ValueError("Integration method name cannot be empty.")
    _INTEGRATION_BUILDERS[canonical] = builder
    _INTEGRATION_ALIASES[canonical] = canonical
    for alias in aliases:
        alias = str(alias).strip().lower().replace("-", "_")
        if not alias:
            raise ValueError("Integration method alias cannot be empty.")
        _INTEGRATION_ALIASES[alias] = canonical
    return builder


def valid_integration_methods():
    """Return the currently registered canonical integration method names."""
    return tuple(sorted(_INTEGRATION_BUILDERS))


def _merge_method_config(current_method, current_kwargs, method, kwargs):
    """Merge inherited/class-body method declarations.

    ``method=None`` means keep the current/inherited method and only update the
    options.  If the method changes, inherited options are intentionally dropped:
    carrying ``max_iter`` from ``derivimplicit`` into ``cnexp`` would be a subtle
    configuration bug.
    """
    kwargs = dict(kwargs or {})
    if method is None:
        current_kwargs.update(kwargs)
        return current_method, current_kwargs

    method = _canonical_method_name(method)
    if method != current_method:
        current_kwargs = {}
    current_kwargs.update(kwargs)
    return method, current_kwargs


def build_integration_func(
    states,
    assigned,
    derivative,
    method,
    eliminate=None,
    diffusion=None,
    **method_kwargs,
):
    """Build the integration function for a State subclass."""
    method = _canonical_method_name(method)
    try:
        builder = _INTEGRATION_BUILDERS[method]
    except KeyError as exc:
        valid = ", ".join(valid_integration_methods())
        raise ValueError(
            f"Unknown integration method: {method!r}. Valid methods are: {valid}."
        ) from exc

    if diffusion:
        if method not in {"euler_maruyama", "euler_heun"}:
            raise ValueError(
                "State.DIFFUSION(...) requires State.METHOD('euler_maruyama') "
                "or State.METHOD('euler_heun') in v1. Voltage/cable SDE solvers "
                "are intentionally out of scope."
            )
        return builder(
            states,
            assigned,
            derivative,
            eliminate=eliminate,
            diffusion=diffusion,
            **method_kwargs,
        )

    if method in {"euler_maruyama", "euler_heun"}:
        return builder(
            states,
            assigned,
            derivative,
            eliminate=eliminate,
            diffusion=(),
            **method_kwargs,
        )

    return builder(
        states,
        assigned,
        derivative,
        eliminate=eliminate,
        **method_kwargs,
    )


register_integration_method("cnexp", build_cnexp, aliases=("rush_larsen", "rl"))
register_integration_method(
    "derivimplicit",
    build_derivimplicit,
    aliases=("deriv_implicit", "implicit", "backward_euler", "be"),
)
register_integration_method(
    "bufferimplicit",
    build_bufferimplicit,
    aliases=(
        "buffer",
        "implicit_buffer",
        "ca_buffer",
        "calcium_buffer",
        "calciumbuffer",
    ),
)
register_integration_method(
    "linearimplicit",
    build_linearimplicit,
    aliases=(
        "linear_implicit",
        "implicitlinear",
        "implicit_linear",
        "affineimplicit",
        "affine_implicit",
        "sparse",
        "be_linear",
        "backward_euler_linear",
        "linear_be",
    ),
)
register_integration_method(
    "euler_maruyama",
    build_euler_maruyama,
    aliases=("em", "sde", "euler-maruyama", "ito"),
)
register_integration_method(
    "euler_heun",
    build_euler_heun,
    aliases=("eh", "stochastic_heun", "stratonovich", "euler-heun"),
)

register_integration_method(
    "rosenbrock",
    build_rosenbrock1,
    aliases=(
        "rosenbrock1",
        "rosenbrock_euler",
        "rosenbrock-euler",
        "linearlyimplicit",
        "linearly_implicit",
        "linearized_implicit",
        "linearly_implicit_euler",
        "linimplicit",
        "semiimplicit",
        "semi_implicit",
    ),
)


class State(Parameterized):
    """
    Helper mixin for declaring per-compartment state variables and their dynamics.

    Define subclasses inside a :class:`Mechanism` and register them with
    :meth:`Mechanism.STATE`. Use uppercase classmethods (``STATE``, ``DERIVATIVE``,
    ``KINETIC``, ``ASSIGNED``, ``BUFFER``, ``GLOBAL``, ``RANGE``) at class
    definition time to declare state variables, ODEs/kinetics, per-compartment
    parameters, and auxiliary buffers. Override lowercase hooks to implement
    behavior:

    - ``initial(self, v)``: populate buffers/states once at initialization.
    - ``breakpoint(self, v, states=None)``: compute ASSIGNED/intermediates each step;
      may return a dict mapping ASSIGNED names to values.
    - ``inf(self, v)``: return steady-state values for states (used for init).
    - ``calc_q10(self)``: optional temperature scaling when ``has_q10=True``.
    """

    _state_buffers = set()
    _state_buffers_declarations = []

    _state = set()
    _state_declarations = []

    _derivative = set()
    _derivative_declarations = []

    _kinetic = set()
    _kinetic_declarations = []

    _diffusion = set()
    _diffusion_declarations = []

    _assigned = set()
    _assigned_declarations = []

    has_q10 = False

    # Backward-compatible public class attribute.  Existing definitions such as
    # ``method = "derivimplicit"`` continue to work.  Prefer ``State.METHOD`` for
    # new code so solver-specific options can be declared alongside the method.
    method = "cnexp"
    method_kwargs = {}

    _method = "cnexp"
    _method_kwargs = {}
    _method_declarations = []

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called automatically whenever a class
        inherits from Parameterized.
        """
        # Call the parent's __init_subclass__ WITHOUT our custom kwargs,
        # as the base 'object' class does not accept them.
        super().__init_subclass__(**kwargs)

        # Start with a fresh dictionary for the new class's parameters.
        new_state = set()
        new_buffers = set()
        new_derivative = set()
        new_assigned = set()
        new_kinetic = set()
        new_diffusion = set()
        new_method = "cnexp"
        new_method_kwargs = {}

        # Walk MRO in reverse to build up params from parent to child.
        for base in reversed(cls.__mro__):
            if "_state" in base.__dict__:
                new_state.update(base._state)
            if "_state_buffers" in base.__dict__:
                new_buffers.update(base._state_buffers)
            if "_derivative" in base.__dict__:
                new_derivative.update(base._derivative)
            if "_kinetic" in base.__dict__:
                new_kinetic.update(base._kinetic)
            if "_diffusion" in base.__dict__:
                new_diffusion.update(base._diffusion)
            if "_assigned" in base.__dict__:
                new_assigned.update(base._assigned)

            if "_method" in base.__dict__:
                new_method, new_method_kwargs = _merge_method_config(
                    new_method,
                    new_method_kwargs,
                    base.__dict__["_method"],
                    base.__dict__.get("_method_kwargs", {}),
                )
            elif "method" in base.__dict__:
                new_method, new_method_kwargs = _merge_method_config(
                    new_method,
                    new_method_kwargs,
                    base.__dict__["method"],
                    base.__dict__.get("method_kwargs", {}),
                )

        if State._state_declarations:
            for s_list in State._state_declarations:
                new_state.update(s_list)
            State._state_declarations = []

        if State._state_buffers_declarations:
            for b_list in State._state_buffers_declarations:
                new_buffers.update(b_list)
            State._state_buffers_declarations = []

        if State._derivative_declarations:
            for d_list in State._derivative_declarations:
                new_derivative.update(d_list)
            State._derivative_declarations = []

        if State._kinetic_declarations:
            for k_list in State._kinetic_declarations:
                new_kinetic.update(k_list)
            State._kinetic_declarations = []

        if State._diffusion_declarations:
            for d_list in State._diffusion_declarations:
                new_diffusion.update(d_list)
            State._diffusion_declarations = []

        if State._assigned_declarations:
            for a_list in State._assigned_declarations:
                new_assigned.update(a_list)
            State._assigned_declarations = []

        if State._method_declarations:
            for method_name, method_kwargs in State._method_declarations:
                new_method, new_method_kwargs = _merge_method_config(
                    new_method, new_method_kwargs, method_name, method_kwargs
                )
            State._method_declarations = []

        new_method = _canonical_method_name(new_method)
        if new_method not in _INTEGRATION_BUILDERS:
            valid = ", ".join(valid_integration_methods())
            raise ValueError(
                f"Unknown integration method for State {cls.__name__}: "
                f"{new_method!r}. Valid methods are: {valid}."
            )

        cls._state = list(new_state)
        cls._state_buffers = new_buffers
        cls._derivative = new_derivative
        cls._kinetic = new_kinetic
        cls._diffusion = new_diffusion
        cls._assigned = list(new_assigned)
        cls._method = new_method
        cls._method_kwargs = dict(new_method_kwargs)

        # Public/introspection aliases.
        cls.method = cls._method
        cls.method_kwargs = dict(cls._method_kwargs)

    def __init__(
        self,
        celsius,
        diameters,
        key,
        shape,
        shape_f,
        additional_parameters=None,
        **kwargs,
    ):
        if not self._state:
            raise ValueError(
                f"State {self.__class__.__name__} has no state variables defined."
                "Use State.STATE(<state vars>) in State implementation to define them."
            )
        super().__init__(
            shape, shape_f, additional_parameters=additional_parameters, **kwargs
        )
        self._name = self.__class__.__name__
        self.key = key

        self.register_buffer("celsius", celsius)
        self.register_buffer("diam", diameters)

        _derivative = list(self._derivative)

        cinfo = None

        if self._kinetic:
            _kinetic = list(self._kinetic)
            deriv_list, _, cinfo, _, _ = kinetic_to_derivatives(self._state, _kinetic)
            _derivative.extend(deriv_list)

        for b in self._state_buffers:
            self.register_buffer(b, torch.tensor(0.0))

        self.include_q10_in_comp_graph = kwargs.get("include_q10_in_comp_graph", False)

        # Integration-method configuration is fixed at State-subclass definition
        # time.  Copy it onto the instance for introspection and to keep generated
        # solver compilation monomorphic.  Instance kwargs may still override
        # legacy global options such as ``pade``.
        self.method = self.__class__._method
        self.method_kwargs = dict(self.__class__._method_kwargs)
        if "pade" in kwargs:
            self.method_kwargs["pade"] = kwargs["pade"]
        else:
            self.method_kwargs.setdefault("pade", False)

        ifunc = build_integration_func(
            self._state,
            self._assigned,
            _derivative,
            self.method,
            eliminate=cinfo,
            diffusion=self._diffusion,
            **self.method_kwargs,
        )
        setattr(self, "solve", MethodType(ifunc, self))

        self._sde_rng_names = ()
        if self.method in {"euler_maruyama", "euler_heun"}:
            self._sde_rng_names = tuple(f"{state}_dW_rng" for state in self._state)
            for rng_name in self._sde_rng_names:
                if not hasattr(self, rng_name):
                    setattr(
                        self,
                        rng_name,
                        RNGModule(None, shape_p=self.shape_p, shape_f=self.shape_f),
                    )

    def populate_parameter_buffers(self, random_generation=None):
        super().populate_parameter_buffers(random_generation=random_generation)
        if self.has_q10:
            if self.include_q10_in_comp_graph:
                self.q10 = self.calc_q10
            else:
                self.q10 = self.return_q10_cache
                self.register_buffer("q10_cache", self.calc_q10())

    @staticmethod
    def to_column(tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a 2D tensor to a column vector (2D tensor with one column).
        """
        return tensor.view(-1, 1)

    def from_column(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a column vector (2D tensor with one column) back to its original shape.
        """
        return tensor.view(*self.shape)

    def initialize(self, v):
        self.initial(v)
        return

    def return_q10_cache(self):
        return self.q10_cache

    def set(self, key: str, value):
        p = getattr(self, key)
        if isinstance(p, torch.Tensor):
            p.data = torch.as_tensor(value, dtype=p.data.dtype, device=p.device)

    @staticmethod
    def STATE(*args):
        """
        Declare state variables for the State subclass.

        Parameters
        ----------
        *args : str
            Names of state variables advanced by the integrator.
        """
        State._state_declarations.append(args)

    @staticmethod
    def BUFFER(*args):
        """
        Declare auxiliary per-compartment buffers.

        Buffers are allocated per instance and typically populated in
        :meth:`initial`; they are not evolved by the ODE solver.

        Parameters
        ----------
        *args : str
            Buffer names to allocate.
        """
        State._state_buffers_declarations.append(args)

    @staticmethod
    def DERIVATIVE(*args):
        """
        Declare ODEs for state variables using symbolic strings.

        Parameters
        ----------
        *args : str
            Derivative expressions like ``"m' = (minf - m) / tau"``.
        """
        State._derivative_declarations.append(args)

    @staticmethod
    def KINETIC(*args):
        """
        Declare kinetic/Markov schemes between states.

        Parameters
        ----------
        *args : str
            Kinetic expressions like ``"~ a <-> b (alpha, beta)"``.
        """
        State._kinetic_declarations.append(args)

    @staticmethod
    def DIFFUSION(*args):
        """Declare diffusion coefficients for Euler-Maruyama state SDEs.

        Each declaration has the form ``"x = sigma"`` and represents the
        multiplicative noise coefficient. ``State.METHOD("euler_maruyama")``
        interprets this as an Itô SDE. ``State.METHOD("euler_heun")`` interprets
        it as a Stratonovich SDE. Voltage-equation noise is intentionally handled
        by future stochastic voltage/cable integrators.
        """
        State._diffusion_declarations.append(args)

    @staticmethod
    def ASSIGNED(*args):
        """
        Declare computed per-compartment variables used in derivatives.

        State ASSIGNED variables form an explicit contract with
        :meth:`breakpoint`: each declared name should be computed there and
        returned in the breakpoint dictionary. For persistent auxiliary storage
        that does not participate in this return contract, use
        :meth:`BUFFER`. Mechanism-level auxiliary storage is declared with
        ``Mechanism.BUFFER(...)``.

        Parameters
        ----------
        *args : str
            Names of ASSIGNED variables to be set in :meth:`breakpoint`.
        """
        State._assigned_declarations.append(args)

    @staticmethod
    def METHOD(method=None, **kwargs):
        """Declare the integration method for this State subclass.

        Parameters
        ----------
        method : str, optional
            Integration method name.  Currently registered methods include
            ``"cnexp"``, ``"derivimplicit"``, ``"bufferimplicit"``,
            ``"linearimplicit"``/``"sparse"``, ``"rosenbrock"``,
            ``"euler_maruyama"`` for Itô State SDEs, and ``"euler_heun"`` for
            Stratonovich State SDEs. Passing ``None`` leaves the inherited method
            unchanged and only updates method kwargs.
        **kwargs
            Method-specific options captured at class-definition time and passed
            to the generated solver builder.  For example::

                State.METHOD("derivimplicit", max_iter=4, line_search=False)

            or::

                State.METHOD("cnexp", pade=True)
        """
        if method is not None:
            method = _canonical_method_name(method)
        State._method_declarations.append((method, dict(kwargs)))

    def breakpoint(self, v, states):
        """
        Compute ASSIGNED/intermediate values for this state at the breakpoint.

        Override in subclasses; may return a dict mapping ASSIGNED names to
        values. Called each step before derivatives are evaluated.
        """
        return {}

    def advance(self, v, dt, states):
        if self.method == "euler_heun":
            return self.solve(v, dt, states)
        return self.solve(dt, **self.breakpoint(v, states), **states)

    def _sde_randn_like(self, state_name: str, like: torch.Tensor) -> torch.Tensor:
        """Detached standard-normal increment for State SDE solvers."""
        rng_name = f"{state_name}_dW_rng"
        rng = getattr(self, rng_name)
        if isinstance(rng, RNGModule):
            rng.init(like.device)
        with torch.no_grad():
            return rng.randn(tuple(like.shape), device=like.device, dtype=like.dtype)

    def init_rng(self):
        super().init_rng()
        for rng_name in getattr(self, "_sde_rng_names", ()):
            rng = getattr(self, rng_name)
            if isinstance(rng, RNGModule):
                rng.init(self._init_device)

    def reset_rng(self):
        super().reset_rng()
        for rng_name in getattr(self, "_sde_rng_names", ()):
            rng = getattr(self, rng_name)
            if isinstance(rng, RNGModule):
                rng.reset()

    def initial(self, v):
        """
        Hook invoked during initialization to populate buffers/states.

        Override to set buffers declared via :meth:`BUFFER` or to customize
        state initialization (may depend on morphology such as ``self.diam``).
        """
        pass

    def inf(self, v):
        """
        Return steady-state values for state variables at voltage ``v``.

        Used during initialization unless overridden by the caller.
        """
        return {}

    def calc_q10(self):
        """
        Optional Q10 scaling helper when ``has_q10=True``.

        Override to return a temperature-dependent multiplicative factor used
        by ``self.q10()`` inside kinetics. Defaults to 1.0.
        """
        return 1.0

    @classproperty
    def code(cls):
        """
        Returns the source code of the mechanism.
        This is useful for debugging and introspection.
        """
        source_code = inspect.getsource(cls)
        return textwrap.dedent(source_code)
