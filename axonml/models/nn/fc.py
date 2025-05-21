from torch import nn


class fully_connected(nn.Module):
    """
    Fully connected layer for axon models.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.
    """

    def __init__(self, hidden_dims, nonlinearity='Sigmoid', bias=True):
        self.net = nn.Sequential()

        dim = 1
        for i in range(len(hidden_dims)):
            self.net.append(nn.Linear(dim, hidden_dims[i], bias=bias))
            self.net.append(getattr(nn, nonlinearity)())
            dim = hidden_dims[i]
        self.net.append(nn.Linear(dim, 1, bias=bias))
        self.net.append(getattr(nn, nonlinearity)())

    def forward(self, x):
        """
        Forward pass through the fully connected layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the fully connected layer.
        """
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)