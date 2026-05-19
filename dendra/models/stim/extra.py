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
    return torch.einsum("can,cat->tan", s, t).contiguous()


def op_sc(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("an,at->tan", s, t).contiguous()


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
        self.fields = []
        self.waveforms = []

        self.is_precomputed = False

        if precomputed is not None:
            self.register_buffer("precomputed", precomputed)
            self.is_precomputed = True
            return

        for field, waveform in field_waveform_tuples:
            if not isinstance(field, torch.Tensor):
                raise TypeError(
                    f"Field must be a torch.Tensor, got {type(field)} instead."
                )
            if not isinstance(waveform, Waveform):
                raise TypeError(
                    f"Waveform must be a Waveform instance, got {type(waveform)} instead."
                )

            self.fields.append(field)
            self.waveforms.append(waveform)

        self.device = None
        self.dtype = None

        fields = torch.stack(self.fields, dim=0)
        self.register_buffer("fields", fields)
        self.register_buffer("waveform_stacked", torch.zeros((self.n_fields, 1)))

        self.register_buffer("idx", torch.zeros(1, dtype=torch.long))
        self.register_buffer("increment", torch.ones(1, dtype=torch.long))

    def set_device_dtype(self, model):
        self.device = model.device()
        self.dtype = model.dtype()

        self.fields = self.fields.to(device=self.device, dtype=self.dtype)
        self.waveforms = [
            waveform.to(device=self.device, dtype=self.dtype)
            for waveform in self.waveforms
        ]

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
        if self.is_precomputed:
            e = self.precomputed[..., self.idx]
            self.idx += self.increment
            return e
        t = self.waveform_stacked[..., self.idx]
        e = torch.einsum("i...,i...->i...", self.fields, t)
        self.idx += self.increment
        return e

    def initialize(self, model, t):
        self.idx.zero_()
        self.set_device_dtype(model)
        t = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        if not self.is_precomputed:
            self.waveform_stacked = torch.stack(
                [waveform(t) for waveform in self.waveforms], dim=0
            )  # shape (model.shape[:], n_timepoints)
