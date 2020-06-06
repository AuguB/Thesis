import itertools
import pickle
import torch
from builder import build_px_samples
from config import project_folder
from marginalizingflow import marginalizingFlow
import numpy as np
import matplotlib.pyplot as plt


folder = "/home/guus/Uni/AI/Thesis/code/bscthesis/Stijn/flows/NormalizingFlows/Runs/run_2020-06-04 16:32:41.224270"
loss_dict = pickle.load(open(folder+"/loss_dict.p","rb"))
model_state_dict_dict = pickle.load(open(folder+"/model_dict.p","rb"))
param_dict = pickle.load(open(folder+"/param_dict.p","rb"))
model_dict = {}

dists = param_dict["dists"]
epsilons = param_dict["epsilons"]
reps = [ i for i in range(param_dict["n_reps"])]
model_tups = itertools.product(dists, epsilons, reps)
loss_tups = itertools.product(dists, epsilons)


for tup in model_tups:
    N = 1 if tup[0] == "square_gaussian" else 2
    model_dict[tup] = marginalizingFlow(N, tup[1],8)
    model_dict[tup].load_state_dict(model_state_dict_dict[tup])
    model_dict[tup].eval()

# Plot the losses
fig, ax = plt.subplots(1,len(dists), figsize=(len(dists)*5,5))

for d, dist in enumerate(dists):
    for e, eps in enumerate(epsilons):
        if len(dists) == 1:
            ax.plot(loss_dict[(dist,eps)].mean(axis=0))
            ax.set_title(dist)
        else:
            ax[d].plot(loss_dict[(dist,eps)].mean(axis=0))
            ax[d].set_title(dist)
plt.show()

# # Sum the inverse errors
# for d, dist in enumerate(dists):
#     data = build_px_samples(500, distribution=dist)
#     for e, eps in enumerate(epsilons):
#         for r, rep in enumerate(reps):
#             marg_flow = model_dict[(dist, eps, rep)]
#             norm_flow = marg_flow.normalizingFlow
#             data_padded, eps_log_prob = marg_flow.add_epsilon(data)
#             log_prob_marg, transformed = norm_flow.forward(data_padded)
#             inverse = norm_flow.inverse(transformed)
#             diff = (inverse-data_padded).abs().sum()
#             print(diff)
            # perfs[d,e,r] = -torch.mean(log_prob_marg).detach().numpy()

# PLot the performance
perfs = np.zeros((len(dists), len(epsilons),len(reps)))
for d, dist in enumerate(dists):
    data = build_px_samples(20, distribution=dist)
    for e, eps in enumerate(epsilons):
        for r, rep in enumerate(reps):
            flow = model_dict[(dist, eps, rep)]
            log_prob_marg, transformed = flow.forward(data, marginalize=True)
            perfs[d,e,r] = -torch.mean(log_prob_marg).detach().numpy()
            # if len(transformed)

fig, ax = plt.subplots(1, len(dists),figsize=(len(dists)*5,5))
perfs = np.mean(perfs, axis = 2)
for d, dist in enumerate(dists):
    if len(dists) == 1:
        ax.plot(epsilons, perfs[d])
    else:
        ax[d].plot(epsilons, perfs[d])
plt.show()

# for d, dist in enumerate(dists):
#     for e, eps in enumerate(epsilons):
#         for r, rep in enumerate(reps):
#             flow = model_dict[dist,eps,r]
#             show_backward(flow,"")