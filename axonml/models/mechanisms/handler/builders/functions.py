from .utils import indent

def parse_args(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"{key}: torch.nn.Module")
    return ", ".join(result)


def parse_ions(ion_names) -> str:
    result = []
    for key in ion_names:
        result.append(f"{key}_ion: torch.nn.Module")
    return ", ".join(result)


def parse_all_states(mechanisms) -> str:
    result = []
    for mech in mechanisms:
        for state in mech.DE:
            result.append(f"'{mech._name}.{state}'")
    s = ", ".join(result)
    return s


def parse_defaults(ions) -> str:
    res = []
    for ion in ions:
        res.append(f"self.{ion}_ion.immediate_update_e()")
    return "\n    ".join(res)


def parse_assignments(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"self.{key} = {key}")
    return "\n    ".join(result)


set_buffer_template = """
self.{mech}.{v} = self.{ion}_ion.{v}
for _, s in self.{mech}.DE.items():
  s.{v} = self.{ion}_ion.{v}
"""


set_ion_write_c_buffer_template = """
self.{mech}.{v} = self.{ion}_ion.{v}
"""


set_diam_buffer_template = """
self.{mech}.diam.set_(diameters)
"""


def parse_set_buffers(mechanisms, ions):
    out = []
    for m in mechanisms:
        for ion in ions:
            if ion in m.read_ion:
                for v in m.read_ion[ion]:
                    out.append(set_buffer_template.format(mech=m._name, v=v, ion=ion))
            if ion in m.write_ion_c:
                for v in m.write_ion_c[ion]:
                    out.append(
                        set_ion_write_c_buffer_template.format(
                            mech=m._name, v=v, ion=ion
                        )
                    )
        out.append(set_diam_buffer_template.format(mech=m._name))
    out = "\n".join(out)
    return indent(out, 1)


def parse_advance(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"self.{key}._advance(v, dt)")
    s = "\n    ".join(result)
    return s


def assign_post_advance(mechanisms, ions) -> str:
    out = []
    for m in mechanisms:
        for ion in ions:
            if ion in m.write_ion_c:
                for v in m.write_ion_c[ion]:
                    out.append(f"self.{ion}_ion.{v} = self.{m._name}.{v}")
    for m in mechanisms:
        for ion in ions:
            if ion in m.read_ion:
                for v in m.read_ion[ion]:
                    if v == f"{ion}i" or v == f"{ion}o":
                        out.append(f"self.{m._name}.{v} = self.{ion}_ion.{v}")
                        for k, _ in m.DE.items():
                            out.append(f"self.{m._name}.DE['{k}'].{v} = self.{ion}_ion.{v}")
    out = "\n    ".join(out)
    return out


def mech_detach(mechanisms):
    result = []
    for m in mechanisms:
        result.append(f"self.{m._name}.detach()")
    return "\n    ".join(result)


def ion_detach(ions):
    result = []
    for ion in ions:
        result.append(f"self.{ion}_ion.detach()")
    return "\n    ".join(result)


def parse_breakpoint(mechanisms):
    ret = []
    for m in mechanisms:
        ret.append(f"self.{m._name}.breakpoint(v)")
    return "\n    ".join(ret)


def parse_ion_advance(ions) -> str:
    if ions is None:
        return ""
    result = []
    for ion in ions:
        result.append(f"self.{ion}_ion.advance(temp)")
    s = "\n    ".join(result)
    return s


def parse_ion_init(ions) -> str:
    if ions is None:
        return ""
    result = []
    for ion in ions:
        result.append(f"self.{ion}_ion.initialize(temp)")
    s = "\n    ".join(result)
    return s


def parse_init_buffers(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"self.{key}._init_buffers_s(v_init)")
    s = "\n    ".join(result)
    return s


def parse_all_states(mechanisms) -> str:
    result = []
    for mech in mechanisms:
        for state in mech.DE:
            result.append(f"'{mech._name}.{state}'")
    s = ", ".join(result)
    return s
