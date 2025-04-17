import torch
import axonml as ax


class Population(torch.jit.ScriptModule):
    """
    A Population is a collection of neurons that can be simulated together.
    It is a subclass of torch.jit.ScriptModule, which allows for JIT compilation
    of the model for performance optimization.

    Parameters
    ----------
    model : class
        The neuron model class to be used for the population.
    diameters : list of float
        List of axon diameters in micrometers.
    L : float, optional
        Length of the cells in micrometers. Default is 13.0.
    **kwargs : dict
        Additional parameters to be passed to the model class.
    """

    def __init__(self, model, diameters, L=13.0, **kwargs):
        super(Population, self).__init__()
        self.n_neurons = len(diameters)
        kwargs['dx'] = L
        kwargs['method'] = "euler"
        with ax.ctx(NETWORK=1): self.model = model(diameters, L=L, **kwargs)

    @torch.jit.script_method
    def step_no_intra(self, cmdt, dt, temp):
        v = self.model.v
        self.model.mech.advance(v, dt, temp)
        irev = self.model.mech.irev()
        gtot = self.model.mech.gtot(v)
        self.model.v = self._advance_implicit(v, irev, gtot, cmdt)
        return
    
    @torch.jit.script_method
    def step_intra(self, cmdt, dt, temp, intra):
        v = self.model.v
        self.model.mech.advance(v, dt, temp)
        irev = self.model.mech.irev()
        gtot = self.model.mech.gtot(v)
        self.model.v = self._advance_implicit_intra(v, irev, gtot, cmdt, intra)
        return

    @property
    def v(self):
        """
        Returns the membrane potential of the neurons in the population.
        """
        return self.model.v
    
    @torch.jit.script_method
    def _advance_implicit(self, v, irev, gtot, cmdt):
        """
        Advances the membrane potential of the neurons in the population
        using an implicit method.
        """
        return (cmdt * v + irev) / (cmdt + gtot)
    
    @torch.jit.script_method
    def _advance_implicit_intra(self, v, irev, gtot, cmdt, intra):
        """
        Advances the membrane potential of the neurons in the population
        using an implicit method with intracellular current.
        """
        return (cmdt * v + irev + intra) / (cmdt + gtot)