import math
from typing import List, Tuple, Optional, Dict, Callable
import re
import itertools

import torch
from torch import Tensor

from axonml import trained
from .callbacks import CallbackList, Callback
from .backend import Backend as A
from .mixins import Parameterized
from .mechanisms.core import Mechanism, validate
from .mechanisms.declarations import PARAMETER
from .mechanisms.handler.handler import build_handler
from .mechanisms.handler.ions import build_ion
from .mechanisms.compiler import compile_mechanism


def get_unique_keys(list_of_dicts):
    """Gets all unique keys from a list of dictionaries.

    Args:
        list_of_dicts: A list of dictionaries.

    Returns:
        A set of unique keys.
    """

    unique_keys = set()
    for dictionary in list_of_dicts:
        unique_keys.update(dictionary.keys())
    return unique_keys


def follows_pattern(base_pattern, target_string):
    regex_pattern = (
        r"\b"
        + r"\b.*?\b".join(re.escape(part) for part in base_pattern.split("."))
        + r"\b"
    )
    return re.search(regex_pattern, target_string) is not None


def matches_any_pattern(base_patterns, target_string):
    for base_pattern in base_patterns:
        regex_pattern = (
            r"\b"
            + r"\b.*?\b".join(re.escape(part) for part in base_pattern.split("."))
            + r"\b"
        )
        if re.search(regex_pattern, target_string):
            return True
    return False


@torch.jit.interface
class HandlerInterface:
    def initialize(self, v, v_init, area, temp) -> None:
        pass

    def advance(self, v, dt) -> None:
        pass

    def i_intra(self, v, intra) -> torch.Tensor:
        pass

    def i(self, v) -> torch.Tensor:
        pass

    def update(self, temp) -> None:
        pass

    def get(self, mech: str, state: str) -> torch.Tensor:
        pass

    def set(self, name: str, value: float) -> None:
        pass

    def all_states(self) -> List[str]:
        pass


class SymmetricConv1D(torch.nn.Conv1d):
    def forward(self, x):
        if self.training:
            weight_ = (self.weight + torch.flip(self.weight, [-1])) / 2
        else:
            weight_ = self.weight
        return self._conv_forward(x, weight_, self.bias)


class Axon(Parameterized, torch.jit.ScriptModule):
    """Base 1D fiber class."""

    def __init__(self, temp=37.0, v_init=-80.0):
        super().__init__()
        self.temp = temp
        self.v_init = v_init
        self.mech: HandlerInterface = None

        self._m_list = []
        self._m_name = []
        self._m_curr = {}

        self._ion_read = {}
        self._ion_write = {}
        self._ion_write_c = {}

        self._all_read = {}
        self._all_write = {}
        self._all_write_c = {}

        self._ion_style = {}

        self.post_initialize_hooks: List[Callable] = []

        self._caches = {}

        # solver stuff
        weight = [1.0, -2.0, 1.0]
        self.ssd = SymmetricConv1D(
            2, 1, 3, bias=False, padding="same", padding_mode="reflect"
        )
        self.ssd.weight.data = torch.tensor([weight, weight]).reshape(1, 2, 3)
        for p in self.ssd.parameters():
            p.requires_grad = False

        self.register_buffer("v", torch.tensor([v_init]))
        self.initialized: bool = False

        # -- constants --
        self.pi = torch.nn.Parameter(torch.tensor(math.pi), requires_grad=False)
        self.eval()

    def unfreeze(self, *names):
        if not names:
            for p in self.parameters():
                p.requires_grad = True
        else:
            for n, p in self.named_parameters():
                if matches_any_pattern(names, n):
                    print(f"Unfreezing {n}")
                    p.requires_grad = True

    def freeze(self, *names):
        if not names:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for n, p in self.named_parameters():
                if matches_any_pattern(names, n):
                    print(f"Freezing {n}")
                    p.requires_grad = False

    def register_post_initialize_hook(self, fn: Callable):
        self.post_initialize_hooks.append(fn)

    @torch.jit.export
    def n(self) -> int:
        return self.v.shape[0]

    def device(self):
        return self.ssd.weight.device

    def dtype(self):
        return self.ssd.weight.dtype

    def insert(self, mechanism: Mechanism, ic=None, **kwargs):
        validate(mechanism)
        # m = mechanism(self.temp, ic=ic, **kwargs)

        m = compile_mechanism(mechanism, self.temp, ic=ic, **kwargs)

        self._m_list.append(m)
        self._m_name.append(mechanism.__name__)

        for k, v in mechanism._currents.items():
            self._m_curr.setdefault(k, {}).update({mechanism.__name__: v})

        for k, v in mechanism._read_ion.items():
            self._ion_read.setdefault(k, {}).update({mechanism.__name__: v})

        for k, v in mechanism._write_ion.items():
            self._ion_write.setdefault(k, {}).update({mechanism.__name__: v})

        for k, v in mechanism._write_ion_c.items():
            self._ion_write_c.setdefault(k, {}).update({mechanism.__name__: v})

        if len(self._ion_write_c) > 1:
            raise ValueError(
                f"Multiple ion channels {self._ion_write_c.keys()} writing concentrations are not supported."
            )

    def ion_style(self, ion, c_style, e_style, einit, eadvance, cinit):
        self._ion_style[ion] = (c_style, e_style, einit, eadvance, cinit)

    def get_ion_style(self, ion):
        if ion in self._ion_style:
            return self._ion_style[ion]
        return self.calc_ion_style(ion)

    def c_is_written(self, ion):
        d = self._ion_write_c.get(ion, {})
        return bool(d)

    def c_is_read(self, ion):
        d = self._ion_read.get(ion, {})
        if not d:
            return False
        check = list(itertools.chain(*d.values()))
        return f"{ion}i" in check or f"{ion}o" in check

    def e_is_read(self, ion):
        d = self._ion_read.get(ion, {})
        if not d:
            return False
        return f"e{ion}" in list(itertools.chain(*d.values()))

    def calc_ion_style(self, ion):
        c_is_written = self.c_is_written(ion)
        c_is_read = self.c_is_read(ion)
        e_is_read = self.e_is_read(ion)

        if c_is_written:
            if e_is_read:
                return (3, 2, 1, 1, 1)
            return (3, 0, 0, 0, 1)
        if c_is_read:
            if e_is_read:
                return (1, 2, 1, 0, 0)
            return (1, 0, 0, 0, 0)
        if e_is_read:
            return (0, 1, 0, 0, 0)
        return (0, 0, 0, 0, 0)

    def build(self):
        all_ions = get_unique_keys([self._ion_read, self._ion_write, self._ion_write_c])

        _ion_write = {f"i{k}": v for k, v in self._ion_write.items()}
        self._m_curr.update(_ion_write)

        ions = {}
        for ion in all_ions:
            ion_write_c = self._ion_write_c.get(ion, {})
            ion_read = self._ion_read.get(ion, {})
            ion_style = self.get_ion_style(ion)
            ions[ion] = build_ion(
                ion, self._m_list, self._m_name, ion_read, ion_write_c, *ion_style
            )
        self.mech = build_handler(
            self._m_list,
            self._m_name,
            self._m_curr,
            self.temp,
            ions,
        )

    def area_(self, diameters):
        raise NotImplementedError()

    def ra_(self, diameters):
        raise NotImplementedError()

    def cm_(self, area):
        return self.cm * area

    def run(
        self,
        ve: Tensor,
        diameters: Tensor,
        dt: float = None,
        intra: Optional[Tensor] = None,
        callbacks: List[Callback] = None,
        reinit: bool = False,
    ):
        with_intra = intra is not None

        self.v_init_c = torch.tensor(
            self.v_init, device=self.device(), dtype=self.dtype()
        )
        self.temp_c = torch.tensor(self.temp, device=self.device(), dtype=self.dtype())

        with torch.set_grad_enabled(self.training):
            device = self.device()

            dt = dt if dt is not None else A.dt
            dt = torch.tensor(dt, device=device)

            if callbacks:
                for c in callbacks:
                    c.dt = dt

            ve = torch.as_tensor(ve, device=device)
            diameters = torch.as_tensor(diameters, device=device)

            if (not self.initialized) or reinit:
                self.v = torch.full_like(ve[0], self.v_init)
                self.area_c = self.area_(diameters)[:, None, None]
                self.cm_c = self.cm_(self.area_c)
                self.ra_c = self.ra_(diameters)[:, None, None]
                self.initialize(self.v, self.v_init_c, self.area_c, self.temp_c)
                self.post_initialize()
                self.initialized = True

            callbacks = CallbackList(callbacks)
            callbacks.pre_loop_hook(self)

            for i in range(len(ve)):
                if with_intra:
                    self.v = self.step_intra(
                        self.v, ve[i], self.area_c, self.cm_c, self.ra_c, dt, intra[i]
                    )
                else:
                    self.v = self.step_no_intra(
                        self.v,
                        ve[i],
                        self.area_c,
                        self.cm_c,
                        self.ra_c,
                        dt,
                    )
                callbacks.post_step_hook(self)

    def post_initialize(self):
        for h in self.post_initialize_hooks:
            h(self)

    @torch.jit.script_method
    def initialize(self, v, v_init, area, temp):
        self.mech.initialize(v, v_init, area, temp)

    @torch.jit.script_method
    def dv(self, cm, ra, d2v, ion, dt) -> Tensor:
        """Calculate dv/dt

        Parameters
        ----------
        cm : torch.Tensor
            Node capacitance.
        ra : torch.Tensor
            Internodal resistance.
        d2v : torch.Tensor
            Spatial 2nd difference.
        ion : torch.Tensor
            Ionic current.
        dt : float
            Timestep.

        Returns
        -------
        Tensor
            dv/dt
        """
        dv = dt * (1 / cm) * (((1 / ra) * d2v) - ion)
        return dv

    @torch.jit.script_method
    def step_no_intra(
        self,
        v,
        ve,
        area,
        cm,
        ra,
        dt,
    ) -> Tensor:
        # -- 2nd diff --
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)

        # -- update gvs --
        self.mech.advance(v, dt)

        # -- calculate ionic current --
        i_ion = self.mech.i(v) * area

        # -- update vm --
        dv = self.dv(cm, ra, d2v, i_ion, dt)
        v = v + dv

        return v

    @torch.jit.script_method
    def step_intra(self, v, ve, area, cm, ra, dt, intra) -> Tensor:
        # -- 2nd diff --
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)

        # -- update gvs --
        self.mech.advance(v, dt)

        # -- calculate ionic current --
        i_ion = self.mech.i(v) * area - intra

        # -- update vm --
        dv = self.dv(cm, ra, d2v, i_ion, dt)
        v = v + dv

        return v

    @torch.jit.script_method
    def get_state(self, s: str) -> Tensor:
        if s == "v":
            return self.v
        mech, state = s.split(".")
        return self.mech.get(mech, state)

    def load(self, state_dict):
        if state_dict in trained:
            state_dict = torch.load(
                trained[state_dict], map_location=self.device(), weights_only=True
            )
        elif isinstance(state_dict, str):
            state_dict = torch.load(
                state_dict, map_location=self.device(), weights_only=True
            )
        self.load_state_dict(state_dict, strict=False)
        return self

    def compile(self, nodes=16, axons=1):
        ve = torch.ones(1, axons, 1, nodes, device=self.device())
        d = 10 * torch.ones(axons, device=self.device())
        for _ in range(5):
            self.run(ve, d, reinit=True)
        self.initialized = False
        return self

    def all_states(self) -> List[str]:
        out = ["v"]
        return out + self.mech.all_states()

    @torch.jit.export
    def set(self, key: str, value: float):
        self.mech.set(key, value)

    @torch.jit.export
    def cache(self, name: str = None):
        if name is None:
            name = "latest"
        self._caches[name] = self.state_dict()

    @torch.jit.export
    def restore(self, name: str = None):
        if name is None:
            name = "latest"
        self.load_state_dict(self._caches[name])
        self.initialized = True


class Unmyelinated(Axon):
    PARAMETER(
        membrane={
            "cm": 1e-3,  # mF / cm2
            "rhoa": 35.4,  # ohm-cm
        }
    )

    def __init__(self, dx=10.0, temp=37, v_init=-80):
        super().__init__(temp, v_init)
        self.dx: float = dx

    def area_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        return self.pi * (diameters / 10000) * dx

    def ra_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        radii = diameters / 20000
        return (self.rhoa * dx) / (self.pi * (radii**2))


class Myelinated(Axon):
    PARAMETER(
        axon_d={
            "axond1": 0.0187623,
            "axond2": 4.787487e-01,
            "axond3": 1.203613e-01,
        },
        node_d={
            "noded1": 6.303781e-03,
            "noded2": 2.070544e-01,
            "noded3": 5.339006e-01,
        },
        delta_x={
            "deltax1": -8.215284e00,
            "deltax2": 2.724201e02,
            "deltax3": -7.802411e02,
        },
        membrane={
            "cm": 10e-3,
            "rhoa": 70.0,  # ohm-cm
        },
    )

    def area_(self, diameters):
        lengths = torch.ones_like(diameters) / 10000
        return self.pi * self.nodeD(diameters) * lengths  # cm2

    def ra_(self, diameters):
        radii = diameters / 20000  # radius in cm
        rhoa = self.rhoa * self.rhoa_scale(diameters)
        return (rhoa * self.deltax(diameters)) / (self.pi * (radii**2))

    def rhoa_scale(self, diameters):
        return 1 / ((self.axonD(diameters) / diameters) ** 2)

    def axonD(self, diameters):
        axond = self.axond1 * diameters**2 + self.axond2 * diameters + self.axond3
        return axond

    def deltax(self, diameters):
        deltax = self.deltax1 * diameters**2 + self.deltax2 * diameters + self.deltax3
        return deltax / 10000

    def nodeD(self, diameters):
        noded = self.noded1 * diameters**2 + self.noded2 * diameters + self.noded3
        return noded / 10000
