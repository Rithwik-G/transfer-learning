import matplotlib.pyplot as plt
from utils.utils import load, save
import os
import numpy as np
from utils.utils import Results
# import scienceplots
# plt.style.use('science')

plt.figure(figsize=(8, 6))

storage_path = "/ocean/projects/phy240020p/rgupta9/transfer_learning/"

# files = ['lsst/lsst_ft_lim.pkl', 'lsst/lsst_direct.pkl'] # 'lsst/lsst_ft_all.pkl'

files = ['bts/bts_ft_output.pkl', 'bts/bts_direct.pkl']#, 'bts/bts_ft_all.pkl', 'bts/bts_ft_last_two.pkl']
# names = ['Transfer Learning', 'Direct Training']

for ind, file in enumerate(files):

    metric = 'macro'

    result = load(os.path.join(storage_path, file))
    res = result.performance
    limits = list(res.keys())
    perf = []
    err = []
    for limit in limits:
        perf.append(np.mean(res[limit][metric]))
        err.append(np.std(res[limit][metric]))
        # perf.append(np.median(res[limit][metric]))
        # err.append(np.median(np.abs(res[limit][metric] - np.median(res[limit][metric]))))

    plt.errorbar(limits, perf, yerr=err, fmt='o', capsize=5, capthick=2, elinewidth=2, label=result.title)# names[ind])

# plt.xscale('log')
plt.legend(fontsize=15)
plt.title("BTS Performance", fontsize=20)
plt.xlabel("Training Samples", fontsize=15)
plt.ylabel("Classification AUROC", fontsize=15)
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/bts_final_macro.pdf", bbox_inches='tight')
plt.show()
plt.clf()
assert(False)

best = 'lsst/lsst_ft_lim.pkl'
res = load(os.path.join(storage_path, best)).performance

limits = list(res.keys())
by_class = {cl: [] for cl in res[limits[0]]['class_based'].keys()}
for limit in limits:
    for cl in res[limit]['class_based'].keys():
        by_class[cl].append(np.mean(res[limit]['class_based'][cl]))

colors = ['r', 'g', 'y', 'b', 'purple', 'orange', 'gray', 'k', 'm', 'c', 'brown', 'olive']
for ind, cl in enumerate(by_class.keys()):
    err = [np.std(res[limit]['class_based'][cl]) for limit in limits]
    perf = by_class[cl]
    plt.errorbar(limits, perf, yerr=err, fmt='o', capsize=3, capthick=1, elinewidth=1, label=cl, c=colors[ind])

plt.legend(fontsize=10)
plt.title("LSST Class-Based Performance", fontsize=20)
plt.xlabel("Training Samples", fontsize=15)
plt.ylabel("Classification AUROC", fontsize=15)
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/lsst_by_class.pdf", bbox_inches='tight')
plt.show()

# m = 'macro'
# fname = "lsst_ft_perf.pkl"


# res = load(os.path.join(storage_path, fname))


# limits = list(res.keys())
# perf = []
# err = []
# for limit in limits:
#     perf.append(np.mean(res[limit][m]))
#     err.append(np.std(res[limit][m]))

# plt.figure(figsize=(8, 6))
# plt.errorbar(limits, perf, yerr=err, fmt='o', capsize=5, capthick=2, elinewidth=2, label='Transfer Learning')

# fname = "bts_direct_perf_5.pkl"
# res = load(os.path.join(storage_path, fname))
# limits = list(res.keys())
# perf = []
# err = []
# for limit in limits:
#     perf.append(np.mean(res[limit][m]))
#     err.append(np.std(res[limit][m]))

# plt.errorbar(limits, perf, yerr=err, fmt='o', capsize=5, capthick=2, elinewidth=2, label='Direct Training')
