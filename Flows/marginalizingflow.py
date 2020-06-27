import torch
from torch import nn
from torch.distributions import MultivariateNormal
from builder import build_flow, build_mnist_flow


class marginalizingFlow(nn.Module):
    def __init__(self, N, M, n_layers=4,mnist= False):
        super(marginalizingFlow, self).__init__()
        self.N = N  # Dimension of original data
        self.M = M  # Dimension of epsilon
        self.Q = max(N + M, 2)  # Dimension of the flow (minimum two)
        if not mnist:
            self.normalizingFlow = build_flow(self.Q, n_layers)
        else:
            self.normalizingFlow = build_mnist_flow(self.Q, n_layers)
        if self.M > 0:
            self.eps = MultivariateNormal(
                loc=torch.zeros(self.M),
                covariance_matrix=torch.diag(torch.ones(self.M)))  # The distribution to sample epsilon

    def forward(self, A, marginalize=False, n_samples=200):
        if not marginalize:
            A_epsilon, _ = self.add_epsilon(A)
            return self.normalizingFlow(A_epsilon)
        else:
            with torch.no_grad():
                # An object to store the log likelihoods for many different epsilons
                log_probs = torch.zeros((A.shape[0], n_samples))
                for s in range(n_samples):
                    A_epsilon, epsilon_log_prob = self.add_epsilon(A)
                    log_prob, transformed = self.normalizingFlow(A_epsilon)
                    log_probs[:, s] = log_prob - epsilon_log_prob
                mean_log_probs = self.logmean(log_probs)
                return mean_log_probs, transformed  # Returns only the last transformation

    def add_epsilon(self, A):
        if self.N + self.M == 1:
            return torch.cat([A, A], 1), 0  # If there is only one column, duplicate
        elif self.M > 0:
            sample = self.eps.sample((A.shape[0],))
            return torch.cat([A, sample], 1), self.eps.log_prob(sample)  # Get new samples from epsilon
        else:
            return A,0

    # The log sum exp trick
    def logmean(self, log_probs):

        max_logprob, ind = torch.max(log_probs, dim=1)
        normed = log_probs - max_logprob.repeat((log_probs.shape[1], 1)).T
        linear = torch.exp(normed)
        mean = torch.mean(linear, dim=1)
        log = torch.log(mean)
        unnormed = log + max_logprob
        return unnormed

        # linear = torch.exp(log_probs)
        # mean = torch.mean(linear)
        # log = torch.log(mean)
        # return log

    # Only inverts, does not provide normalization
    def inverse(self, A):
        return self.normalizingFlow.inverse(A)