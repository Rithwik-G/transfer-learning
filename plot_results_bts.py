import matplotlib.pyplot as plt
from utils.utils import load, save
import os
import numpy as np
from utils.utils import Results
from matplotlib.ticker import FuncFormatter
# import scienceplots
# plt.style.use('science')

plt.figure(figsize=(8, 6))

storage_path = "/ocean/projects/phy240020p/rgupta9/transfer_learning"



files = ['bts/bts_ft_last_two', 'bts/bts_direct_five']#, 'bts/bts_ft_full_five.pkl', 'bts/bts_ft_last_two.pkl']
names = ['Transfer Learning', 'Direct Training']

# files = ['bts/bts_ft_output_five1', 'bts/bts_ft_all', 'bts/bts_ft_last_two']
# names = ['Output Only', 'Fully Unfrozen', 'Last Two Layers']

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

obj_per_day = 9513/2346.1470601996407 # times observed to get days
obj_per_month = obj_per_day * 30.44 # times observed to get months

for ind, file in enumerate(files):

    metric = 'macro'

    result = load(os.path.join(storage_path, file + '.pkl'))
    res = result.performance
    
    limits = list(res.keys())

    perf = []
    err = []
    for limit in limits:
        if (limit > 4000):
            limits = limits[:-2]
            break
        
        perf.append(np.median(res[limit][metric]))
        err.append(np.median(np.abs(res[limit][metric] - np.median(res[limit][metric]))))
    perf = np.array(perf)
    err = np.array(err)
    plt.plot(limits, perf, marker='.', label= names[ind], color=colors[ind])
    plt.fill_between(limits, perf - err, perf + err, alpha=0.1, color=colors[ind])

# Add a top axis that shows the number of months
def samples_to_months(x):
    return x / obj_per_month

def months_to_samples(x):
    return x * obj_per_month



ax = plt.gca()
secax = ax.secondary_xaxis('top', functions=(samples_to_months, months_to_samples))
secax.set_xlabel("Months of Data", fontsize=12)
secax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
# ax.set_ylim(0.2, 0.95)
plt.legend(fontsize=15)
# plt.title("BTS Performance", fontsize=20)
plt.xlabel("Training Samples", fontsize=15)
plt.ylabel("Classification AUROC", fontsize=15)
os.makedirs("final_figures", exist_ok=True)
plt.grid()
plt.savefig("final_figures/bts.pdf", bbox_inches='tight')
plt.show()
plt.clf()


total_samples = 4000
by_class = dict(zip(['SLSN-I', 'SN II', 'SN Ia', 'SN Ib/c', 'TDE'], [  29,  762, 2982,  215,    9]))


# Add a top axis that scales to the by_class counts
comparison = files # ['bts/bts_ft_output_five1', 'bts/bts_direct_five1']
# Compare each class from the two files in 'comparison'
results = [load(os.path.join(storage_path, f + '.pkl')).performance for f in comparison]
limits = list(results[1].keys())
classes = list(results[1][limits[0]]['class_based'].keys())

fig, axs = plt.subplots(len(classes), 1, figsize=(8, 4 * len(classes)), sharex=True)

for idx, cl in enumerate(classes):
    for res_idx, res in enumerate(results):
        perf = [np.median(res[limit]['class_based'][cl]) for limit in limits]
        err = [np.median(np.abs(res[limit]['class_based'][cl] - np.median(res[limit]['class_based'][cl]))) for limit in limits]
        perf = np.array(perf)
        err = np.array(err)
        axs[idx].plot(limits, perf, marker='.', label=names[res_idx], color=colors[res_idx])
        axs[idx].fill_between(limits, perf - err, perf + err, alpha=0.1, color=colors[res_idx])
    axs[idx].set_ylabel(f"{cl} AUROC", fontsize=12)
    
    axs[idx].set_ylim(0.1, 0.95)
    axs[idx].grid()
    # axs[idx].set_title(f"BTS: {cl}", fontsize=14)


def make_top_axis(ax, class_name):
    # Get the number of samples for this class
    n = by_class[class_name]
    def scale(x):
        return (x * n / total_samples)
    def inv_scale(x):
        return (x * total_samples / n)

    secax = ax.secondary_xaxis('top', functions=(scale, inv_scale))
    # secax.set_xlabel(f"Training Samples ({class_name})", fontsize=12, labelpad=10)
    # Add the label just below the top axis, inside the plot area
    ax.annotate(f"{class_name} Training Samples", 
                xy=(0.5, 0.93), 
                xycoords='axes fraction', 
                ha='center', va='bottom', 
                fontsize=12)
    secax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    return secax

# Add top axes for each subplot
for idx, cl in enumerate(classes):
    make_top_axis(axs[idx], cl)

axs[-1].set_xlabel("Training Samples", fontsize=15)
axs[-1].legend(fontsize=15)
plt.tight_layout()
plt.savefig("final_figures/bts_cls.pdf", bbox_inches='tight')
plt.show()

assert(False)

res = load(os.path.join(storage_path, best + '.pkl')).performance

limits = list(res.keys())
by_class = {cl: [] for cl in res[limits[0]]['class_based'].keys()}
for limit in limits:
    for cl in res[limit]['class_based'].keys():
        by_class[cl].append(np.median(res[limit]['class_based'][cl]))



for ind, cl in enumerate(by_class.keys()):
    err = [np.median(np.abs(res[limit]['class_based'][cl] - np.median(res[limit]['class_based'][cl]))) for limit in limits]
    perf = by_class[cl]
    # plt.errorbar(limits, perf, yerr=err, fmt='o', capsize=3, capthick=1, elinewidth=1, label=cl, c=colors[ind])
    perf = np.array(perf)
    err = np.array(err)
    plt.plot(limits, perf, marker='.', label= cl, color=colors[ind])
    plt.fill_between(limits, perf - err, perf + err, alpha=0.1, color=colors[ind])

plt.legend(fontsize=10)
plt.title("Transfer Learning Class-Based Performance", fontsize=20)
plt.xlabel("Training Samples", fontsize=15)
plt.ylabel("Classification AUROC", fontsize=15)
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/lsst_by_class_ft.pdf", bbox_inches='tight')
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
