import torch

from .waveform import Waveform


def ve_from_s_t(space, time, n, device, multicontact=False):
    ve_s = torch.as_tensor(space, device=device)
    ve_t = torch.as_tensor(time, device=device)

    if multicontact:
        ve_s = ve_s.expand(-1, n, -1)
        ve_t = ve_t.expand(-1, n, -1)
        einsum = op_mc
    else:
        ve_s = ve_s.expand(n, -1)
        ve_t = ve_t.expand(n, -1)
        einsum = op_sc

    return einsum(ve_s, ve_t)


def op_mc(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("c...n,c...t->t...n", s, t).contiguous()


def op_sc(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...n,...t->t...n", s, t).contiguous()


class Extra(torch.nn.Module):
    """
    Handles extracellular stimulation for models.
    """

    def __init__(
        self,
        field_waveform_tuples: list[tuple[torch.Tensor, Waveform]],
        precomputed=None,
    ):
        """
        Initialize extracellular stimulation handler.

        Parameters
        ----------
        field_waveform_tuples : list of (torch.Tensor, Waveform)
            Each tuple contains a spatial field (shape (n_cell, n_comp)) and a
            temporal waveform (Waveform instance).
        """
        super(Extra, self).__init__()
        self.is_precomputed = precomputed is not None
        self.device = None
        self.dtype = None
        self.register_buffer("idx", torch.zeros(1, dtype=torch.long))
        self.register_buffer("increment", torch.ones(1, dtype=torch.long))

        if precomputed is not None:
            if not isinstance(precomputed, torch.Tensor):
                raise TypeError(
                    "precomputed extracellular stimulation must be a torch.Tensor."
                )
            if precomputed.ndim == 0:
                raise ValueError(
                    "precomputed extracellular stimulation needs a time axis."
                )
            self.register_buffer("precomputed", precomputed)
            self.waveforms = torch.nn.ModuleList()
            return

        if not field_waveform_tuples:
            raise ValueError("Extra requires at least one field/waveform pair.")

        fields = []
        waveforms = []
        for field, waveform in field_waveform_tuples:
            if not isinstance(field, torch.Tensor):
                raise TypeError(
                    f"Field must be a torch.Tensor, got {type(field)} instead."
                )
            if not isinstance(waveform, Waveform):
                raise TypeError(
                    f"Waveform must be a Waveform instance, got {type(waveform)} instead."
                )

            fields.append(field)
            waveforms.append(waveform)

        self.n_fields = len(fields)
        try:
            fields_stacked = torch.stack(fields, dim=0)
        except RuntimeError as exc:
            raise ValueError(
                "All extracellular fields must have the same shape."
            ) from exc
        self.register_buffer("fields", fields_stacked)
        self.waveforms = torch.nn.ModuleList(waveforms)
        self.register_buffer(
            "waveform_stacked",
            torch.zeros((self.n_fields, 1)),
            persistent=False,
        )

    def set_device_dtype(self, model):
        self.device = model.device()
        self.dtype = model.dtype()

        if self.is_precomputed:
            self.precomputed = self.precomputed.to(device=self.device, dtype=self.dtype)
        else:
            self.fields = self.fields.to(device=self.device, dtype=self.dtype)
            self.waveforms.to(device=self.device, dtype=self.dtype)

    @classmethod
    def from_precomputed(cls, precomputed_extracellular: torch.Tensor):
        """
        Create an Extra instance from precomputed extracellular data.

        Parameters
        ----------
        precomputed_extracellular : torch.Tensor
            Precomputed extracellular data with shape (n_field, n_cell, n_timepoints).

        Returns
        -------
        Extra
            An instance of the Extra class.
        """
        return cls(field_waveform_tuples=[], precomputed=precomputed_extracellular)

    def forward(self):
        position = int(self.idx.item())
        n_time = (
            self.precomputed.shape[-1]
            if self.is_precomputed
            else self.waveform_stacked.shape[-1]
        )
        if position < 0 or position >= n_time:
            raise IndexError(
                f"Extracellular stimulation is exhausted at time index {position}; "
                f"only {n_time} samples were initialized."
            )
        if self.is_precomputed:
            e = self.precomputed[..., position]
            self.idx += self.increment
            return e
        t = self.waveform_stacked[..., position]
        while t.ndim < self.fields.ndim:
            t = t.unsqueeze(-1)
        e = (self.fields * t).sum(dim=0)
        self.idx += self.increment
        return e

    def initialize(self, model, t):
        self.idx.zero_()
        self.set_device_dtype(model)
        t = torch.atleast_1d(torch.as_tensor(t, device=self.device, dtype=self.dtype))
        expected_shape = tuple(model.v.shape)
        spatial_shape = (
            tuple(self.precomputed.shape[:-1])
            if self.is_precomputed
            else tuple(self.fields.shape[1:])
        )
        try:
            broadcast_shape = torch.broadcast_shapes(spatial_shape, expected_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Extracellular field shape {spatial_shape} is not broadcastable "
                f"to model shape {expected_shape}."
            ) from exc
        if broadcast_shape != expected_shape:
            raise ValueError(
                f"Extracellular field shape {spatial_shape} broadcasts beyond "
                f"model shape {expected_shape}."
            )

        if self.is_precomputed:
            if self.precomputed.shape[-1] != t.numel():
                raise ValueError(
                    "Precomputed extracellular stimulation must contain one "
                    f"sample per timepoint; got {self.precomputed.shape[-1]} "
                    f"samples for {t.numel()} timepoints."
                )
        else:
            evaluated = []
            for waveform in self.waveforms:
                value = waveform(t).to(device=self.device, dtype=self.dtype)
                if value.ndim == 0:
                    value = value.expand_as(t)
                if value.shape[-1] != t.numel():
                    raise ValueError(
                        "Waveform output must use time as its last dimension; "
                        f"got shape {tuple(value.shape)} for {t.numel()} timepoints."
                    )
                evaluated.append(value)
            try:
                evaluated = torch.broadcast_tensors(*evaluated)
                self.waveform_stacked = torch.stack(evaluated, dim=0)
            except RuntimeError as exc:
                raise ValueError(
                    "All extracellular waveforms must return compatible shapes."
                ) from exc
