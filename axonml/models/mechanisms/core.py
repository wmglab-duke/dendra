from torch import Module


class Mechanism(Module):

    currents = set()
    states = set()

    def __init__(self) -> None:
        super(Mechanism, self).__init__()

    