import torch
from torch import nn
from torch.distributions import MultivariateNormal

class normalizingFlow(nn.Module):
    def __init__(self, modulelist, Q):
        super(normalizingFlow, self).__init__()
        self.locs = nn.Parameter(torch.zeros(Q), requires_grad=False)
        self.cov = nn.Parameter(torch.diag(torch.ones(Q)),requires_grad=False)
        self.pu = MultivariateNormal(self.locs,self.cov)
        self.parts = nn.ModuleList(modulelist) # a sequence of flows
        self.Q = Q

    def forward(self, A):

        cumm_logdet = torch.zeros_like(A[:, 0])
        for m in self.parts:
            A, logdet = m(A)
            cumm_logdet += logdet
        print(A.device.type)
        print(cumm_logdet.device.type)
        print(self.pu.loc.device.type)
        print(self.pu.covariance_matrix.device.type)
        return self.pu.log_prob(A) + cumm_logdet, A

    def inverse(self, A):
        # TODO implement inverse log_p
        parts = [i for i in self.parts]
        for m in reversed(parts):
            A = m.inverse(A)
        return A

    def self_evaluate(self, data, sample_size = 200):
        with torch.no_grad():
            sample_size = min(data.shape[0], sample_size)
            sample_indices = torch.randperm(data.shape[0])[:sample_size]
            sample = data[sample_indices]
            log_prob, transformed_data = self.forward(sample)
            log_prob = torch.mean(log_prob).detach().numpy()

            mean = torch.mean(transformed_data, dim = 0)
            C = transformed_data-mean
            cov =  (C.T @ C) / (sample_size-1)
            approximate_gaussian = torch.distributions.MultivariateNormal(loc = mean,covariance_matrix=cov)
            true_gaussian = self.pu
            kl_div = torch.distributions.kl_divergence(true_gaussian, approximate_gaussian).detach().numpy()

            return log_prob, kl_div





