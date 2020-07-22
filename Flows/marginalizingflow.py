import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from builder import build_flow


class marginalizingFlow(nn.Module):
    def __init__(self, data_dimensions, auxiliary_dimensions, n_layers=4):
        super(marginalizingFlow, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data_dimensions = data_dimensions  # Dimension of original data
        self.auxiliary_dimensions = auxiliary_dimensions  # Dimension of epsilon
        self.dimension_of_flows = max(data_dimensions + auxiliary_dimensions, 2)  # Dimension of the flow (minimum two)
        self.normalizingFlow = build_flow(self.dimension_of_flows, n_layers)
        self.locs = nn.Parameter(torch.zeros(self.auxiliary_dimensions).to(device), requires_grad=False)
        self.cov = nn.Parameter(torch.diag(torch.ones(self.auxiliary_dimensions).to(device)),
                                requires_grad=False)
        if self.auxiliary_dimensions > 0:

            self.distribution_of_epsilon = MultivariateNormal(self.locs,
                                                              self.cov)

    def forward(self, a, marginalize=False, n_samples=200):
        if (not marginalize) or (self.auxiliary_dimensions == 0):
            a_with_epsilon, _ = self.add_epsilon(a)
            return self.normalizingFlow(a_with_epsilon)
        else:
            # An object to store the log likelihoods for many different epsilons
            log_prob_buffer = torch.zeros((a.shape[0], n_samples))
            for s in range(n_samples):
                a_with_epsilon, log_prob_of_epsilon = self.add_epsilon(a)
                log_prob, transformed = self.normalizingFlow(a_with_epsilon)
                log_prob_buffer[:, s] = log_prob.clone().detach() - log_prob_of_epsilon
            mean_log_probs = self.logmean(log_prob_buffer).clone().detach()
            return mean_log_probs, transformed  # Returns only the last transformation

    def add_epsilon(self, a):
        if self.data_dimensions + self.auxiliary_dimensions == 1:
            return torch.cat([a, a], 1), 0  # Flows require at least 2-D data
        elif self.auxiliary_dimensions > 0:
            epsilon_sample = self.distribution_of_epsilon.sample((a.shape[0],))
            return torch.cat([a, epsilon_sample], 1), self.distribution_of_epsilon.log_prob(
                epsilon_sample)  # Get new samples from epsilon
        else:
            return a, 0

    # The log sum exp trick
    def logmean(self, log_probs):
        device = log_probs.device
        np_logprobs = log_probs.detach().cpu().numpy().astype(np.float128)
        log_prob_max = np.max(np_logprobs, axis=1,keepdims=True)
        max_array = np.repeat(log_prob_max,log_probs.shape[1], axis = 1)
        normalized_log_probs = np_logprobs - max_array
        normalized_probs = np.exp(normalized_log_probs)
        mean_of_normalized_probs = np.mean(normalized_probs, axis=1)
        mean_of_normalized_log_probs = np.log(mean_of_normalized_probs)
        mean_of_log_probs = mean_of_normalized_log_probs + log_prob_max
        return torch.from_numpy(mean_of_log_probs.astype(np.float64)).to(device)

    # Only inverts, does not provide normalization
    def inverse(self, a):
        return self.normalizingFlow.inverse(a)
