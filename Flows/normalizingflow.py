import torch
from torch import nn
from torch.distributions import MultivariateNormal

class normalizingFlow(nn.Module):
    def __init__(self, modulelist, Q):
        super(normalizingFlow, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mean_of_base_distribution = torch.zeros(Q).to(device)
        self.covariance_of_base_distribution = torch.diag(torch.ones(Q)).to(device)
        self.base_distribution = MultivariateNormal(self.mean_of_base_distribution, self.covariance_of_base_distribution)
        self.component_flows = nn.ModuleList(modulelist) # a sequence of flows
        self.dimension_of_flows = Q

    def forward(self, a):
        cumm_logdet = torch.zeros_like(a[:, 0])
        for m in self.component_flows:
            a, logdet = m(a)
            cumm_logdet += logdet
        return self.base_distribution.log_prob(a) + cumm_logdet, a

    def inverse(self, a):
        # TODO implement inverse log_p
        list_of_components = [i for i in self.component_flows]
        for component in reversed(list_of_components):
            a = component.inverse(a)
        return a





