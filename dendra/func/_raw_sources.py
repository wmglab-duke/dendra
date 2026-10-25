"""Cold source locations and ordered routes for raw parameter packets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class _RawSourceBinding:
    path: object
    canonical: str
    schema: tuple | None = None


@dataclass(frozen=True)
class _RawSourceInventory:
    """Describe packets and execution routes without retaining source values.

    Packet rows retain their own order, independently of canonical alias
    groups. Discovery, classification and source resolution belong to each
    consumer; this inventory never reads modules or materializes tensors.
    """

    parameters: dict
    constants: dict
    routes: dict
    _packet_aliases: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_packet_aliases",
            tuple(
                (kind, key, binding.canonical)
                for kind, bindings in (
                    ("parameter", self.parameters),
                    ("constant", self.constants),
                )
                for key, binding in bindings.items()
                if key != binding.canonical
            ),
        )

    @classmethod
    def from_paths(
        cls,
        parameter_paths,
        constant_paths,
        *,
        routes,
        parameter_canonical=None,
        constant_canonical=None,
    ):
        parameter_canonical = {} if parameter_canonical is None else parameter_canonical
        constant_canonical = {} if constant_canonical is None else constant_canonical
        return cls(
            {
                key: _RawSourceBinding(path, parameter_canonical.get(key, key))
                for key, path in parameter_paths.items()
            },
            {
                key: _RawSourceBinding(path, constant_canonical.get(key, key))
                for key, path in constant_paths.items()
            },
            {
                scope: (dict(parameters), dict(constants))
                for scope, (parameters, constants) in routes.items()
            },
        )

    def aliases(self, kind, scope=None):
        group = self.routes.get(scope)
        return {} if group is None else group[{"parameter": 0, "constant": 1}[kind]]

    def packet_aliases(self):
        return self._packet_aliases

    def schema_items(self, kind):
        for key, binding in getattr(self, f"{kind}s").items():
            yield key, binding.schema

    def with_schema(self, kind, values):
        """Capture one caller-resolved frame at its original schema boundary."""
        name = f"{kind}s"
        bindings = getattr(self, name)
        return replace(
            self,
            **{
                name: {
                    key: replace(
                        bindings[key],
                        schema=(tuple(value.shape), value.dtype, value.device),
                    )
                    for key, value in values.items()
                }
            },
        )
