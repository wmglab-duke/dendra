import torch


def to_param(val):
    return torch.nn.Parameter(torch.tensor(val), requires_grad=False)


class Parameterized(torch.nn.Module):
    
    _params = None
    
    def __init__(self):
        super(Parameterized, self).__init__()
        self.instantiate_parameters()
        self.is_cuda = False
        
    def instantiate_parameters(self):
        if self.__class__._params is not None:
            for name, value in self.__class__._params.items():
                if isinstance(value, dict):
                    setattr(self, name, [])
                    for pname, pval in value.items():
                        setattr(self, pname, to_param(pval))
                        getattr(self, name).append(getattr(self, pname))
                else:
                    setattr(self, name, to_param(value))

    def cuda(self):
        self.is_cuda = True
        return super().cuda()
    
    def cpu(self):
        self.is_cuda = False
        return super().cpu()
                    
    def device(self) -> str:
        return 'cuda' if self.is_cuda else 'cpu'