from ._mechanism import Mechanism


class Synapse(Mechanism):
    def net_receive(self, weights):
        """
        Update the synaptic conductance based on incoming spikes.
        This method should be overridden by specific synapse implementations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")