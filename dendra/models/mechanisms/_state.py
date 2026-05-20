import inspect
import textwrap
from types import MethodType

import torch

from dendra.models.parametric import Parameterized

from ._cnexp import build_cnexp
from ._derivimplicit import build_derivimplicit
from ._kinetic import kinetic_to_derivatives
from ._mechanism import classproperty


def build_integration_func(
    states, assigned, derivative, method, eliminate=None, pade=False
):
    """
    Build the integration function for the states and assigned variables.
    """
    if method == "cnexp":
        return build_cnexp(states, assigned, derivative, eliminate=eliminate, pade=pade)
    elif method == "derivimplicit":
        return build_derivimplicit(
            states, assigned, derivative, eliminate=eliminate, pade=pade
        )
    else:
        raise ValueError(
            f"Unknown integration method: {method}. Valid methods are: cnexp, derivimplicit."
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

    _assigned = set()
    _assigned_declarations = []

    has_q10 = False
    method = "cnexp"

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

        # Walk MRO in reverse to build up params from parent to child
        for base in reversed(cls.__mro__):
            # We look for a _params attribute defined directly on the base
            if "_state" in base.__dict__:
                new_state.update(base._state)
            if "_state_buffers" in base.__dict__:
                new_buffers.update(base._state_buffers)
            if "_derivative" in base.__dict__:
                new_derivative.update(base._derivative)
            if "_kinetic" in base.__dict__:
                new_kinetic.update(base._kinetic)
            if "_assigned" in base.__dict__:
                new_assigned.update(base._assigned)

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

        if State._assigned_declarations:
            for a_list in State._assigned_declarations:
                new_assigned.update(a_list)
            State._assigned_declarations = []

        cls._state = list(new_state)
        cls._state_buffers = new_buffers
        cls._derivative = new_derivative
        cls._kinetic = new_kinetic
        cls._assigned = list(new_assigned)

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

        pade = kwargs.get("pade", False)
        self.include_q10_in_comp_graph = kwargs.get("include_q10_in_comp_graph", False)

        ifunc = build_integration_func(
            self._state, self._assigned, _derivative, self.method, cinfo, pade
        )
        setattr(self, "solve", MethodType(ifunc, self))

    def populate_parameter_buffers(self):
        super().populate_parameter_buffers()
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
            Derivative expressions like ``\"m' = (minf - m) / tau\"``.
        """
        State._derivative_declarations.append(args)

    @staticmethod
    def KINETIC(*args):
        """
        Declare kinetic/Markov schemes between states.

        Parameters
        ----------
        *args : str
            Kinetic expressions like ``\"~ a <-> b (alpha, beta)\"``.
        """
        State._kinetic_declarations.append(args)

    @staticmethod
    def ASSIGNED(*args):
        """
        Declare computed per-compartment variables used in derivatives.

        Parameters
        ----------
        *args : str
            Names of ASSIGNED variables to be set in :meth:`breakpoint`.
        """
        State._assigned_declarations.append(args)

    def breakpoint(self, v, states):
        """
        Compute ASSIGNED/intermediate values for this state at the breakpoint.

        Override in subclasses; may return a dict mapping ASSIGNED names to
        values. Called each step before derivatives are evaluated.
        """
        return {}

    def advance(self, v, dt, states):
        return self.solve(dt, **self.breakpoint(v, states), **states)

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
