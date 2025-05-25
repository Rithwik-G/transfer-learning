import pickle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

class Results:
    def __init__(self, performance, title):
        self.performance = performance
        self.title = title


def save(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)



def get_auroc_cls(y_pred, y_true):

    y_true_bin = label_binarize(y_true, classes=list(range(len(np.unique(y_true)))))

    auroc_per_class = roc_auc_score(y_true_bin, y_pred, average=None)
    
    macro_auroc = roc_auc_score(y_true_bin, y_pred, average='macro')
    micro_auroc = roc_auc_score(y_true_bin, y_pred, average='micro')
    
    return {'by_class': auroc_per_class, 'macro': macro_auroc, 'micro': micro_auroc}

