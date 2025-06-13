import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
avoid_classes = ['other', 'Other', 'ILRT', 'LBV', 'LRN', 'Ca-rich', 'nova', 'SLSN-II'] # comment last two
class_map = {
    'SN Ibn': 'SN Ib/c',
    'SN Icn': 'SN Ib/c',
    'SN Ic-BL': 'SN Ib/c',
    'SN Ia-91bg': 'SN Ia',
    'SN Ia-91T': 'SN Ia',
}

def load_bts_data(pad=-1):

    target = load("../../data/BTS/target.pkl")
    x_data = load("../../data/BTS/x_data.pkl")
    host_galaxy_info = load("../../data/BTS/host_galaxy_info.pkl")


    lengths = []
    for lc in x_data:
        lengths.append(len(lc))

    ntimesteps = np.max(lengths)
    print("Max timesteps is:", ntimesteps)
    ntimesteps=max(ntimesteps, pad)

    # Pad for TF masking layer
    for ind in range(len(x_data)):
        x_data[ind] = x_data[ind][:, [0, 3, 1, 2]]
        x_data[ind][:, 0] = x_data[ind][:, 0] / 10000
        x_data[ind] = np.pad(x_data[ind], ((0, ntimesteps - len(x_data[ind])), (0, 0)))

    y_data_norm = []
    x_data_norm = []
    host_gal = []

    for i in range(len(target)):
        if (target[i] in class_map):
            target[i] = class_map[target[i]]
        if (target[i] in avoid_classes):
            pass
        elif True: # (target[i] in ['SN Ia', 'SN II', 'SN Ib/c']):
            x_data_norm.append(x_data[i])
            y_data_norm.append(target[i])
            host_gal.append(host_galaxy_info[i])

    # One-hot Encoding

    enc = OneHotEncoder(handle_unknown='ignore')

    print(np.unique(y_data_norm, return_counts=True))

    y_data_norm = enc.fit_transform(np.array(y_data_norm).reshape(-1, 1)).todense()

    X_train, X_test, host_gal_train, host_gal_test, y_train, y_test = train_test_split(x_data_norm, host_gal, y_data_norm, random_state = 45, test_size = 0.2)
    X_train, X_val, host_gal_train, host_gal_val, y_train, y_val = train_test_split(X_train, host_gal_train, y_train, random_state = 45, test_size = 0.2)

    class_weights = {i : 0 for i in range(y_train.shape[1])}

    for value in y_train:
        class_weights[np.argmax(value)]+=1

    for id in class_weights.keys():
        class_weights[id] = len(y_train) / class_weights[id]

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    X_test = np.array(X_test)
    y_test = np.array(y_test)


    host_gal_train = np.array(host_gal_train)
    host_gal_test = np.array(host_gal_test)
    host_gal_val = np.array(host_gal_val)

    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)
    y_test = np.squeeze(y_test)

    ordered_class_names = enc.categories_

    return X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, ordered_class_names, class_weights