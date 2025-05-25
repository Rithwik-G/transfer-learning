import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder


from collections import defaultdict, Counter
import random

def sample(target, freq_dict):
    # Count actual occurrences of each class in target
    class_counts = Counter(target)
    
    # Normalize the frequency distribution
    total_freq = sum(freq_dict.values())
    normalized_freq = {k: v / total_freq for k, v in freq_dict.items()}

    # Determine the maximum total sample size such that no class is overused
    max_possible = float('inf')
    for cls, freq in normalized_freq.items():
        if freq > 0:
            max_possible = min(max_possible, class_counts[cls] / freq)

    # Determine how many samples to take per class
    sample_counts = {
        cls: int(freq * max_possible)
        for cls, freq in normalized_freq.items()
    }

    # Bucket indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(target):
        class_indices[label].append(idx)

    # Sample indices
    sampled_indices = []
    for cls, count in sample_counts.items():
        available = class_indices[cls]
        sampled = random.sample(available, k=count)
        sampled_indices.extend(sampled)

    random.shuffle(sampled_indices)
    return sampled_indices


# Get the mean wavlengths for each filter and then convert to micro meteres
wavelengths = {
    'u': (320 + 400) / (2 * 1000),
    'g': (400 + 552) / (2 * 1000),
    'r': (552 + 691) / (2 * 1000),
    'i': (691 + 818) / (2 * 1000),
    'z': (818 + 922) / (2 * 1000),
    'Y': (950 + 1080) / (2 * 1000),
}

class_map = {
                'SNII-NMF': 'SNII', 
                'SNIc-Templates': 'SNIb', 
                'CART': 'CART', 
                'EB': 'EB', 
                'SNIc+HostXT_V19': 'SNIc', 
                'd-Sct': 'Delta Scuti', 
                'SNIb-Templates': 'SNIb', 
                'SNIIb+HostXT_V19': 'SNIIb', 
                'SNIcBL+HostXT_V19': 'SNIc-BL', 
                'CLAGN': 'AGN', 
                'PISN': 'PISN', 
                'Cepheid': 'Cepheid', 
                'TDE': 'TDE', 
                'SNIa-91bg': 'SNIa-91bg', 
                'SLSN-I+host': 'SLSN-I', 
                'SNIIn-MOSFIT': 'SNIIn', 
                'SNII+HostXT_V19': 'SNII', 
                'SLSN-I_no_host': 'SLSN-I', 
                'SNII-Templates': 'SNII', 
                'SNIax': 'SNIax', 
                'SNIa-SALT3': 'SNIa', 
                'KN_K17': 'KN', 
                'SNIIn+HostXT_V19': 'SNIIn', 
                'dwarf-nova': 'Dwarf Novae', 
                'uLens-Binary': 'uLens', 
                'RRL': 'RR Lyrae', 
                'Mdwarf-flare': 'M-dwarf Flare', 
                'ILOT': 'ILOT', 
                'KN_B19': 'KN', 
                'uLens-Single-GenLens': 'uLens', 
                'SNIb+HostXT_V19': 'SNIb', 
                'uLens-Single_PyLIMA': 'uLens'
            }

anomaly_map = OrderedDict({
                'SNII-NMF': 'common',
                'SNIc-Templates': 'common',
                'CART': 'anomaly',
                'EB': 'anomaly',
                'SNIc+HostXT_V19': 'common',
                'd-Sct': 'anomaly',
                'SNIb-Templates': 'common',
                'SNIIb+HostXT_V19': 'common',
                'SNIcBL+HostXT_V19': 'common',
                'CLAGN': 'common',
                'PISN': 'anomaly',
                'Cepheid': 'anomaly',
                'TDE': 'common',
                'SNIa-91bg': 'common',
                'SLSN-I+host': 'common',
                'SNIIn-MOSFIT': 'common',
                'SNII+HostXT_V19': 'common',
                'SLSN-I_no_host': 'common',
                'SNII-Templates': 'common',
                'SNIax': 'common',
                'SNIa-SALT3': 'common',
                'KN_K17': 'anomaly',
                'SNIIn+HostXT_V19': 'common',
                'dwarf-nova': 'anomaly',
                'uLens-Binary': 'anomaly',
                'RRL': 'anomaly',
                'Mdwarf-flare': 'anomaly',
                'ILOT': 'anomaly',
                'KN_B19': 'anomaly',
                'uLens-Single-GenLens': 'anomaly',
                'SNIb+HostXT_V19': 'common',
                'uLens-Single_PyLIMA': 'anomaly'
            })

ap_map = OrderedDict({
    class_map[key]: anomaly_map[key] for key in anomaly_map
})

anom_classes = []
non_anom_classes = []
classes = []

for key in ap_map:
    classes.append(key)
    if ap_map[key] == 'anomaly':
        anom_classes.append(key)
    else:
        non_anom_classes.append(key)
        
    
print(non_anom_classes)
print(anom_classes)

def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def load_lsst_data(sample_freq=None):
    print("Loading LSST Data")
    target = load("../../ELaSTiCC_Data/processed/target.pkl")

    # Converting the elasticc class to the astrophysical class
    target = [class_map[e_class] for e_class in target]
    x_data = load("../../ELaSTiCC_Data/processed/x_data.pkl")
    host_galaxy_info = load("../../ELaSTiCC_Data/processed/host_galaxy_info2.pkl")
    
    # Cuts number of objects to 13,000 transients max
    np.random.seed(42)

    target = np.array(target)


    # for class_name, counts in zip(*np.unique(target, return_counts=True)):
    #     # Remove the extra samples
    #     if counts > 13000:
    #         indices = np.where(target == class_name)[0]
    #         indices = np.random.choice(indices, counts - 13000, replace=False, )
    #         target = np.delete(target, indices)
    #         x_data = [x for i, x in enumerate(x_data) if i not in indices]
    #         host_galaxy_info = np.delete(host_galaxy_info, indices, axis=0)
    
    
    
    # Get number of observations for all light curves and get max number of timesteps 
    lengths = []
    for lc in x_data:
        lengths.append(len(lc))

    ntimesteps = np.max(lengths)

    for ind in range(len(x_data)):
        x_data[ind] = np.pad(x_data[ind], ((0, ntimesteps - len(x_data[ind])), (0, 0)))
        
    # Split data into anomalous and normal classes and x inputs and y targets

    y_data_anom = []
    y_data_norm = []
    x_data_norm = []
    x_data_anom = []
    host_gal_anom = []
    host_gal = []
    lengths_norm = []
    lengths_anom = []

    for i in range(len(target)):

        if (target[i] in anom_classes):
            x_data_anom.append(x_data[i])
            y_data_anom.append(target[i])
            host_gal_anom.append(host_galaxy_info[i])
            lengths_anom.append(lengths[i])

        else:
            x_data_norm.append(x_data[i])
            y_data_norm.append(target[i])
            host_gal.append(host_galaxy_info[i])
            lengths_norm.append(lengths[i])

    if (sample_freq == None):
        sample_freq = {}
        for cl in np.unique(y_data_norm):
            sample_freq[cl] = 1

    # indices = sample(y_data_norm, sample_freq)
    y_data_norm = np.array(y_data_norm)# [indices]
    x_data_norm = np.array(x_data_norm)# [indices]
    host_gal = np.array(host_gal)# [indices]
    lengths_norm = np.array(lengths_norm)# [indices]

    print('LSST Classes', np.unique(y_data_norm, return_counts=True))
            
            
   # One-hot Encoding

    

    enc = OneHotEncoder(handle_unknown='ignore')

    y_data_norm = enc.fit_transform(np.array(y_data_norm).reshape(-1, 1)).todense()
    print(enc.categories_)

    # Train-validation-test split: 80% training, 10% validation, 10% test 
    X_train, X_test, host_gal_train, host_gal_test, y_train, y_test, lengths_train, lengths_test = train_test_split(x_data_norm, host_gal, y_data_norm, lengths_norm, random_state = 40, test_size = 0.1)
    X_train, X_val, host_gal_train, host_gal_val, y_train, y_val, lengths_train, lengths_val = train_test_split(X_train, host_gal_train, y_train, lengths_train, random_state = 40, test_size = 1/9)
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    lengths_train=np.array(lengths_train)
    lengths_test=np.array(lengths_test)
    lengths_val=np.array(lengths_val)


    host_gal_train = np.array(host_gal_train)
    host_gal_test = np.array(host_gal_test)
    host_gal_val = np.array(host_gal_val)

    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)
    y_test = np.squeeze(y_test)

    X_val = X_val[:, :, [3, 0, 1, 2]]
    X_test = X_test[:, :, [3, 0, 1, 2]]
    X_train = X_train[:, :, [3, 0, 1, 2]]

    class_weights = {i: 0 for i in range(y_train.shape[1])}
    for i in y_train:
        class_weights[np.argmax(i)] += 1
    for i in class_weights.keys():
        class_weights[i] = len(y_train) / class_weights[i]

    ordered_class_names = enc.categories_
    # for i in range(len(enc.categories_[0])):
    #     ordered_class_names.append(enc.categories_[0][i])
    # ordered_class_names = np.array(ordered_class_names)

    return X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, class_weights, ordered_class_names



    
    



