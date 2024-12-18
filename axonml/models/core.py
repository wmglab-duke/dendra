import math
from typing import List, Tuple, Optional, Dict, Callable
import re
import itertools
import warnings

import torch
from torch import Tensor

from tqdm.auto import tqdm

from axonml import trained
from .callbacks import CallbackList, Callback
from .backend import Backend as A
from .mixins import Parameterized
from .mechanisms.core import Mechanism, validate
from .mechanisms.declarations import PARAMETER
from .mechanisms.handler.handler import build_handler
from .mechanisms.handler.ions import build_ion
from .mechanisms.mech_compiler import compile_mechanism


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

    __constants__ = ["method", "n_ax", "n_node", "temp", "v_init"]

    def __init__(self, diameters, n_node: int, temp=37.0, v_init=-80.0, method="rk1"):
        super().__init__()

        self.method_conversion = {
            "rk1": "rk1",
            "rk4": "rk4",
            "dufort-frankel": "df",
        }

        if method not in self.method_conversion:
            raise ValueError(
                f"Invalid method: {method}. Valid methods: {list(self.method_conversion.keys())}"
            )

        # self.pi = torch.nn.Parameter(torch.tensor(math.pi), requires_grad=False)

        self.register_buffer("diam", torch.as_tensor(diameters))
        self.register_buffer("area_c", self.area_(self.diam)[:, None, None])
        self.register_buffer("cm_c", self.cm_(self.area_c))
        self.register_buffer("ra_c", self.ra_(self.diam)[:, None, None])

        self.method = method
        self.n_ax = len(diameters)
        self.n_node = n_node
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

        self.weight_choices = {
            "rk1": [[1.0, -2.0, 1.0], [1.0, -2.0, 1.0]],
            "rk4": [[1.0, -2.0, 1.0], [1.0, -2.0, 1.0]],
            "dufort-frankel": [
                [1.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [1.0, -2.0, 1.0],
            ],
        }

        self.nc = {"rk1": 2, "rk4": 2, "dufort-frankel": 3}

        # solver stuff
        weight = self.weight_choices[method]
        nc = self.nc[method]

        self.ssd = SymmetricConv1D(
            nc, 1, 3, bias=False, padding="same", padding_mode="reflect"
        )
        self.ssd.weight.data = torch.tensor(weight).reshape(1, nc, 3)
        for p in self.ssd.parameters():
            p.requires_grad = False

        self.register_buffer("v", torch.full((self.n_ax, 1, n_node), v_init))
        if method == "dufort-frankel":
            self.register_buffer("v_prev", torch.full((self.n_ax, 1, n_node), v_init))

        self.initialized: bool = False

        # -- constants --
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

        df = self.method == "dufort-frankel"

        with warnings.catch_warnings():
            warnings.simplefilter("once")
            m = compile_mechanism(
                mechanism,
                self.temp,
                self.diam,
                self.n_ax,
                self.n_node,
                ic=ic,
                df=df,
                **kwargs,
            )

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
                ion,
                self.n_ax,
                self.n_node,
                self._m_list,
                self._m_name,
                ion_read,
                ion_write_c,
                *ion_style,
            )
            for m in self._m_list:
                m.register_ion(ions[ion])
                # if m._name in self._ion_write_c.get(ion, {}):
                #    ions[ion].register_write_c(m)

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

    def init_v(self):
        self.v[:] = self.v_init
        if self.method == "dufort-frankel":
            self.v_prev[:] = self.v_init

    def run(
        self,
        ve: Tensor,
        dt: float = None,
        intra: Optional[Tensor] = None,
        callbacks: List[Callback] = None,
        reinit: bool = False,
        progressbar: bool = True,
    ):
        with_intra = intra is not None

        self.v_init_c = torch.tensor(
            self.v_init, device=self.device(), dtype=self.dtype()
        )
        self.temp_c = torch.tensor(self.temp, device=self.device(), dtype=self.dtype())

        method = getattr(self, f"step_no_intra_{self.method_conversion[self.method]}")
        method_intra = getattr(
            self, f"step_intra_{self.method_conversion[self.method]}"
        )

        df = self.method == "dufort-frankel"

        with torch.set_grad_enabled(self.training):
            device = self.device()

            dt = dt if dt is not None else A.dt
            dt = torch.tensor(dt, device=device)

            if callbacks:
                for c in callbacks:
                    c.dt = dt

            ve = torch.as_tensor(ve, device=device)

            if (not self.initialized) or reinit:
                self.init_v()
                self.initialize(self.v, self.v_init_c, self.area_c, self.temp_c)
                self.post_initialize()
                self.initialized = True

            callbacks = CallbackList(callbacks)
            callbacks.pre_loop_hook(self)

            if df:
                s = 2 * dt / self.cm_c
                s2 = s / self.ra_c
            else:
                cm_inv = 1 / self.cm_c
                ra_inv = 1 / self.ra_c

            for i in tqdm(range(ve.shape[0]), desc="Running", disable=not progressbar):
                if df:
                    if with_intra:
                        self.v, self.v_prev = method_intra(
                            self.v,
                            self.v_prev,
                            ve[i],
                            self.area_c,
                            s,
                            s2,
                            dt,
                            intra[i],
                        )
                    else:
                        self.v, self.v_prev = method(
                            self.v,
                            self.v_prev,
                            ve[i],
                            self.area_c,
                            s,
                            s2,
                            dt,
                        )
                else:
                    if with_intra:
                        self.v = method_intra(
                            self.v,
                            ve[i],
                            self.area_c,
                            cm_inv,
                            ra_inv,
                            dt,
                            intra[i],
                        )
                    else:
                        self.v = method(
                            self.v,
                            ve[i],
                            self.area_c,
                            cm_inv,
                            ra_inv,
                            dt,
                        )
                callbacks.post_step_hook(self)

    def post_initialize(self):
        for h in self.post_initialize_hooks:
            h(self)

    @torch.jit.script_method
    def initialize(self, v, v_init, area, temp):
        self.mech.initialize(v, v_init, area, temp)

    def FRK(self, v, ve, area, cm, ra):
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)
        i_ion = self.mech.i(v) * area
        return cm * ((ra * d2v) - i_ion)

    def FRK_intra(self, v, ve, area, cm, ra, intra):
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)
        i_ion = self.mech.i(v) * area - intra
        return cm * ((ra * d2v) - i_ion)

    @torch.jit.script_method
    def step_no_intra_rk1(
        self,
        v,
        ve,
        area,
        cm,
        ra,
        dt,
    ) -> Tensor:
        self.mech.advance(v, dt)
        K1 = self.FRK(v, ve, area, cm, ra)
        v = v + K1 * dt

        return v

    @torch.jit.script_method
    def step_intra_rk1(self, v, ve, area, cm, ra, dt, intra) -> Tensor:
        self.mech.advance(v, dt)
        K1 = self.FRK_intra(v, ve, area, cm, ra, intra)
        v = v + K1 * dt

        return v

    @torch.jit.script_method
    def step_no_intra_rk4(
        self,
        v,
        ve,
        area,
        cm,
        ra,
        dt,
    ) -> Tensor:
        self.mech.advance(v, dt)

        # -- update vm --
        K1 = self.FRK(v, ve, area, cm, ra)
        K2 = self.FRK(v + (dt / 2) * K1, ve, area, cm, ra)
        K3 = self.FRK(v + (dt / 2) * K2, ve, area, cm, ra)
        K4 = self.FRK(v + dt * K3, ve, area, cm, ra)

        v = v + (dt / 6) * (K1 + 2 * K2 + 2 * K3 + K4)

        return v

    @torch.jit.script_method
    def step_intra_rk4(
        self,
        v,
        ve,
        area,
        cm,
        ra,
        dt,
        intra,
    ) -> Tensor:
        self.mech.advance(v, dt)

        # -- update vm --
        K1 = self.FRK_intra(v, ve, area, cm, ra, intra)
        K2 = self.FRK_intra(v + (dt / 2) * K1, ve, area, cm, ra, intra)
        K3 = self.FRK_intra(v + (dt / 2) * K2, ve, area, cm, ra, intra)
        K4 = self.FRK_intra(v + dt * K3, ve, area, cm, ra, intra)

        v = v + (dt / 6) * (K1 + 2 * K2 + 2 * K3 + K4)

        return v

    @torch.jit.script_method
    def step_no_intra_df(self, v, v_prev, ve, area, s, s2, dt) -> Tuple[Tensor, Tensor]:
        self.mech.advance(v, dt)

        # -- 2nd diff --
        x = torch.cat([v, v_prev, ve], dim=1)
        d2v = self.ssd(x)

        # -- calculate ionic current --
        i_ion = self.mech.i(v_prev) * area

        # -- update vm --
        v_new = (v_prev + s2 * d2v - s * i_ion) / (1 + s2 + s * self.mech.gtot() * area)

        return v_new, v

    @torch.jit.script_method
    def step_intra_df(
        self, v, v_prev, ve, area, s, s2, dt, intra
    ) -> Tuple[Tensor, Tensor]:
        self.mech.advance(v, dt)

        # -- 2nd diff --
        x = torch.cat([v, v_prev, ve], dim=1)
        d2v = self.ssd(x)

        # -- calculate ionic current --
        i_ion = self.mech.i(v_prev) * area - intra

        # -- update vm --
        v_new = (v_prev + s2 * d2v - s * i_ion) / (1 + s2 + s * self.mech.gtot() * area)

        return v_new, v

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
        matched, _ = match_state_dict(self.state_dict(), state_dict)
        self.load_state_dict(matched, strict=False)
        return self

    def compile(self, callbacks: List[Callback] = None):
        ve = torch.ones(1, self.n_ax, 1, self.n_node, device=self.device())
        for _ in range(5):
            self.run(ve, callbacks=callbacks, progressbar=False)
        self.initialized = False
        if callbacks:
            for c in callbacks:
                c.reset()
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


def match_state_dict(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Filters state_dict_b to contain only states that are present in state_dict_a.

    Matching happens according to two criteria:
        - Is the key present in state_dict_a?
        - Does the state with the same key in state_dict_a have the same shape?

    Returns
        (matched_state_dict, unmatched_state_dict)

        States in matched_state_dict contains states from state_dict_b that are also
        in state_dict_a and unmatched_state_dict contains states that have no
        corresponding state in state_dict_a.

            In addition: state_dict_b = matched_state_dict U unmatched_state_dict.
    """
    matched_state_dict = {
        key: state
        for (key, state) in state_dict_b.items()
        if key in state_dict_a and state.shape == state_dict_a[key].shape
    }
    unmatched_state_dict = {
        key: state
        for (key, state) in state_dict_b.items()
        if key not in matched_state_dict
    }
    return matched_state_dict, unmatched_state_dict


class Unmyelinated(Axon):
    PARAMETER(
        membrane={
            "cm": 1e-3,  # mF / cm2
            "rhoa": 35.4,  # ohm-cm
        }
    )

    def __init__(self, diameters, L=1.0, dx=10.0, temp=37, v_init=-80, method="rk1"):
        L = L * 1000  # mm -> um
        n_node = L / dx
        n_node = math.ceil(n_node) // 2 * 2 + 1
        self.dx: float = dx
        super().__init__(diameters, n_node, temp, v_init, method)

    def x(self) -> torch.Tensor:
        l = (self.n_node - 1) * self.dx
        return torch.linspace(-l / 2, l / 2, self.n_node)

    def area_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        return torch.pi * (diameters / 10000) * dx

    def ra_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        radii = diameters / 20000
        return (self.rhoa * dx) / (torch.pi * (radii**2))


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
        return torch.pi * self.nodeD(diameters) * lengths  # cm2

    def ra_(self, diameters):
        radii = diameters / 20000  # radius in cm
        rhoa = self.rhoa * self.rhoa_scale(diameters)
        return (rhoa * self.deltax(diameters)) / (torch.pi * (radii**2))

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
