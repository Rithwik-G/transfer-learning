import matplotlib.pyplot as plt
from utils.utils import load, save
import os
import numpy as np
from utils.utils import Results
# import scienceplots
# plt.style.use('science')

plt.figure(figsize=(8, 6))

storage_path = "rgupta9/transfer_learning"

files = [['lsst/lsst_ft_pref_suff'], ['lsst/lsst_ft_output'], ['lsst/lsst_ft_only_output']]
names = ['GRU and Output', 'Output Only', 'Full Model']

colors = [
    "#fc8d62",  # coral
    "#66c2a5",  # teal
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#c5c578",  # yellow
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
    "#8da0cb"   # periwinkle
]

total = 366652
total_time = 1095

for ind, file_list in enumerate(files):

    metric = 'macro'
    combined_res = {}

    # Load and combine results from all files in file_list
    for file in file_list:
        result = load(os.path.join(storage_path, file + '.pkl'))
        res = result.performance
        for limit, value in res.items():
            if limit not in combined_res:
                combined_res[limit] = []
            combined_res[limit].extend(value[metric])

    # Remove limits > 10000
    removed = [limit for limit in combined_res if limit > 10000]
    for limit in removed:
        del combined_res[limit]

    limits = sorted(combined_res.keys())
    removed = [900, 1000]
    for limit in removed:
        if limit in limits:
            limits.remove(limit)
    perf = []
    err = []


    for limit in limits:
        vals = np.array(combined_res[limit])
        perf.append(np.median(vals))
        err.append(np.median(np.abs(vals - np.median(vals))))
    perf = np.array(perf)
    err = np.array(err)
    plt.plot(limits, perf, marker='.', label=names[ind], color=colors[ind])
    plt.fill_between(limits, perf - err, perf + err, alpha=0.1, color=colors[ind])


ax = plt.gca()
def samples_to_time(x):
    return np.array(x) / total * total_time

def time_to_samples(x):
    return np.array(x) * total / total_time


plt.legend(fontsize=15)
plt.xlabel("Training Samples", fontsize=15)
plt.ylabel("Classification AUROC", fontsize=15)
os.makedirs("figures", exist_ok=True)
plt.grid()
plt.savefig("final_figures/lsst_hyper.pdf", bbox_inches='tight')
plt.show()
plt.clf()


best = 'lsst/lsst_ft_pref_suff'
res = load(os.path.join(storage_path, best + '.pkl')).performance

limits = list(res.keys())
removed = [900, 1000]
for limit in removed:
    if limit in limits:
        limits.remove(limit)

by_class = {cl: [] for cl in res[limits[0]]['class_based'].keys()}
for limit in limits:
    for cl in res[limit]['class_based'].keys():
        by_class[cl].append(np.median(res[limit]['class_based'][cl]))



for ind, cl in enumerate(by_class.keys()):
    err = [np.median(np.abs(res[limit]['class_based'][cl] - np.median(res[limit]['class_based'][cl]))) for limit in limits]
    perf = by_class[cl]
    perf = np.array(perf)
    err = np.array(err)
    plt.plot(limits, perf, marker='.', label= cl, color=colors[ind])
    plt.fill_between(limits, perf - err, perf + err, alpha=0.1, color=colors[ind])

plt.legend(fontsize=10, ncol=2)

plt.xlabel("Training Samples", fontsize=15)
plt.ylabel("Classification AUROC", fontsize=15)

os.makedirs("figures", exist_ok=True)
plt.grid()
plt.savefig("final_figures/lsst_cls.pdf", bbox_inches='tight')
plt.show()
