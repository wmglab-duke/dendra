import math
import torch


class Synapse:
    def __init__(self, e):
        self.n_axons = 0
        self.n_nodes = 0
        self.dt = None

        self.e = e
        self.driver = None
        self.weight = None

    def drive(self, driver, weight):
        self.driver = driver
        self.weight = weight
        return self

    def init(self, n_axons, n_nodes, dt, device, dtype):
        self.n_axons = n_axons
        self.n_nodes = n_nodes
        self.dt = dt
        if self.driver is not None:
            self.driver.init(n_axons, n_nodes, device, dtype)
        return self

    def __call__(self, t, v):
        self.register_events(t)
        self.advance()
        return self.i(v)

    def register_events(self, t):
        pass

    def advance(self):
        pass

    def i(self, v):
        pass


class ExpSyn(Synapse):
    def __init__(self, e=0.0, tau=2.0):
        super().__init__(e)
        self.tau = tau

    def init(self, n_axons, n_nodes, dt, device, dtype):
        super().init(n_axons, n_nodes, dt)
        self.dexp = math.exp(-self.dt / self.tau)
        self.g = torch.zeros(n_axons, n_nodes, device=device, dtype=dtype)
        return self

    def register_events(self, t):
        if self.driver is not None:
            self.g += self.driver(t) * self.weight

    def advance(self):
        self.g *= self.dexp

    def i(self, v):
        return 1e-6 * self.g * (v - self.e)


class Exp2Syn(Synapse):
    def __init__(self, e=0.0, tau1=0.1, tau2=10.0):
        super().__init__(e)

        if tau1 / tau2 > 0.9999:
            tau1 = 0.9999 * tau2

        if tau1 / tau2 < 1e-9:
            tau1 = tau2 * 1e-9

        self.tau1 = tau1
        self.tau2 = tau2

        tp = (tau1 * tau2) / (tau2 - tau1) * math.log(tau2 / tau1)
        factor = -math.exp(-tp / tau1) + math.exp(-tp / tau2)
        self.factor = 1 / factor

    def drive(self, driver, weight):
        super().drive(driver, weight)
        self.factor *= weight
        return self

    def init(self, n_axons, n_nodes, dt, device, dtype):
        super().init(n_axons, n_nodes, dt)
        self.dexp1 = math.exp(-self.dt / self.tau1)
        self.dexp2 = math.exp(-self.dt / self.tau2)
        self.A = torch.zeros(n_axons, n_nodes, device=device, dtype=dtype)
        self.B = torch.zeros(n_axons, n_nodes, device=device, dtype=dtype)
        return self

    def register_events(self, t):
        if self.driver is not None:
            inc = self.driver(t) * self.factor
            self.A += inc
            self.B += inc

    def advance(self):
        self.A *= self.dexp1
        self.B *= self.dexp2

    def i(self, v):
        g = self.B - self.A
        return 1e-6 * g * (v - self.e)
