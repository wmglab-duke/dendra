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

    def __init__(self, field_waveform_tuples: list[tuple[torch.Tensor, Waveform]]):
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

        self.n_fields = len(self.fields)
        self.device = None
        self.dtype = None
        self.n_cell = None  # to be set later

        fields = torch.stack(self.fields, dim=0)
        self.register_buffer("fields", fields)
        self.register_buffer("waveform_stacked", torch.zeros((self.n_fields, 1)))

    def set_device_dtype_ncell(self, model):
        self.device = model.device()
        self.dtype = model.dtype()
        self.n_cell = model.np

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
        if precomputed_extracellular.ndim != 3:
            raise ValueError(
                "Precomputed extracellular data must be 3-D (n_field, n_cell, n_timepoints)."
            )

    def init(self, t):
        n_cell = self.n_cell
        device = self.device
        dtype = self.dtype
        if self.fields.shape[1] != n_cell or self.fields.ndim != 2:
            raise ValueError(
                f"Field tensors must have shape (n_field, n_cell, n_compartments), got {self.fields.shape} instead."
            )

        self.fields = self.fields.to(device=device, dtype=dtype)
        self.waveforms = [
            waveform.to(device=device, dtype=dtype) for waveform in self.waveforms
        ]

        t = torch.as_tensor(t, device=self.device, dtype=self.dtype).unsqueeze(0)
        self.waveform_stacked = torch.stack(
            [waveform(t).expand(n_cell, -1) for waveform in self.waveforms], dim=0
        )  # shape (n_field, n_cell, n_timepoints)
