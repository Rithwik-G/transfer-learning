from utils.load_ztf_data import load_ztf_data
from utils.load_lsst_data import load_lsst_data
# from utils.models import build_model, train, train_contextual
import numpy as np
import os
from utils.utils import save, load, get_auroc_cls, Results
from utils.models import transfer
from utils.models import build_model

from astromcad import astromcad

storage_path = "/ocean/projects/phy240020p/rgupta9/transfer_learning"

os.makedirs(storage_path, exist_ok=True)

# res = load(os.path.join(storage_path, 'lsst/lsst_ft_all.pkl'))
# save(os.path.join(storage_path, 'lsst/lsst_ft_all.pkl'), res.performance)

# res = load(os.path.join(storage_path, 'lsst/lsst_ft_lim.pkl'))
# save(os.path.join(storage_path, 'lsst/lsst_ft_lim.pkl'), res.performance)

class Dataset:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, class_weights, ordered_class_names):
        self.ordered_class_names = ordered_class_names[0]
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.host_gal_train = host_gal_train
        self.host_gal_test = host_gal_test
        self.host_gal_val = host_gal_val
        self.class_weights = class_weights





if __name__ == '__main__':
    # Load the lsst data
    X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, ordered_class_names, class_weights, ntimesteps=load_ztf_data()
    ztf = Dataset(X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, class_weights, ordered_class_names)

    X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, class_weights, ordered_class_names=load_lsst_data()
    lsst = Dataset(X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, class_weights, ordered_class_names)


    # print(len(X_train) + len(X_val) + len(X_test))
    # assert(False)
    # for _ in range(1, 6):
    #     model = astromcad.build_model(latent_size=100, ntimesteps=ztf.X_train.shape[1], num_classes=ztf.y_train.shape[1], contextual=0, n_features=ztf.X_train.shape[2])
    #     astromcad.train(model, ztf.X_train, ztf.y_train, ztf.X_val, ztf.y_val, ztf.class_weights, epochs=40)
    #     # Save the model
    #     model_pth = os.path.join(storage_path, f"models/ztf_classifier{i}.weights.h5")
    #     model.save_weights(model_pth)
    
    # class_freq = None
    # X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, class_weights = load_lsst_data(656, sample_freq=class_freq)
    # lsst = Dataset(X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, class_weights)
    print('LSST Class Order', lsst.ordered_class_names)
    print('ZTF Class Order', ztf.ordered_class_names)
    # assert(False)

    def experiment(pre, limits, fname, unfrozen, long_title):
        lsst_direct_perf = {}

        
        
        for i in range(1, 6):
            for limit in limits:
                if (limit not in lsst_direct_perf):
                    lsst_direct_perf[limit] = {'class_based' : {cl : [] for cl in lsst.ordered_class_names}, 'macro' : [], 'micro': []}
                print(f"LSST Direct Training {i} {limit}")
                new_model=None
                
                new_model = build_model(latent_size=100, ntimesteps=lsst.X_train.shape[1], num_classes=lsst.y_train.shape[1], contextual=0, n_features=lsst.X_train.shape[2])
                if (pre):
                    model_pth = os.path.join(storage_path, f"models/ztf_classifier_new{i}.weights.h5")
                    try:
                        # Load weights except for the 'output' layer
                        # Load weights to a temp model with same architecture as ZTF
                        temp_model = build_model(latent_size=100, ntimesteps=ztf.X_train.shape[1], num_classes=ztf.y_train.shape[1], contextual=0, n_features=ztf.X_train.shape[2])
                        temp_model.load_weights(model_pth)
                        # Copy weights for all layers except 'output'
                        for layer, temp_layer in zip(new_model.layers, temp_model.layers):
                            if layer.name != 'output':
                                layer.set_weights(temp_layer.get_weights())
                    except Exception as e:
                        print("Retrained Foundation Model")
                        # Train and save ZTF model if weights not found
                        temp_model = build_model(latent_size=100, ntimesteps=ztf.X_train.shape[1], num_classes=ztf.y_train.shape[1], contextual=0, n_features=ztf.X_train.shape[2])
                        astromcad.train(temp_model, ztf.X_train, ztf.y_train, ztf.X_val, ztf.y_val, ztf.class_weights, epochs=40)
                        temp_model.save_weights(model_pth)
                        # Copy weights for all layers except 'output'
                        for layer, temp_layer in zip(new_model.layers, temp_model.layers):
                            if layer.name != 'output':
                                layer.set_weights(temp_layer.get_weights())

                    for layer in new_model.layers:
                        if (layer.name in unfrozen):
                            layer.trainable = True
                        else:
                            layer.trainable = False

                print(new_model.summary())
                
                val = limit//10
                train = limit - val
                astromcad.train(new_model, lsst.X_train[:train], lsst.y_train[:train], lsst.X_val[:val], lsst.y_val[:val], lsst.class_weights, epochs=40)

                lsst_pred = new_model.predict(lsst.X_test)
                lsst_true = [np.argmax(lsst.y_test[i]) for i in range(len(lsst.y_test))]

                performance = get_auroc_cls(lsst_pred, lsst_true)
                print(f"LSST Performance: {i} {limit} ", performance)


                lsst_direct_perf[limit]['micro'].append(performance['micro'])
                lsst_direct_perf[limit]['macro'].append(performance['macro'])

                by_class_performance = dict(zip(lsst.ordered_class_names, performance['by_class']))
                for cl in lsst.ordered_class_names:
                    lsst_direct_perf[limit]['class_based'][cl].append(by_class_performance[cl])
                
                res = Results(lsst_direct_perf, long_title)
                if (fname):
                    save(os.path.join(storage_path, f"lsst/{fname}.pkl"), res)
        print(f"{fname} Performance: ", lsst_direct_perf)


    limits = [100, 200, 300, 400, 500, 600]#, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    unfrozen = ['gru_1', 'gru_2', 'dense_3', 'latent', 'output']

    # experiment(pre=False, limits=limits, fname='lsst_direct_new', unfrozen=unfrozen,  long_title='LSST Direct Training')
    # experiment(pre=False, limits=limits, fname='lsst_direct5', unfrozen=None, long_title='LSST Direct Training')
    # experiment(pre=True, limits=limits, fname='lsst_ft_all_new', unfrozen=unfrozen, long_title='LSST Fine Tuned Fully Unfrozen')
    # experiment(pre=True, limits=limits, fname='lsst_ft_pref_suff_new', unfrozen=['gru_1', 'gru_2', 'latent', 'output'], long_title='LSST Fine Tuned Pref Suff')
    # experiment(pre=True, limits=limits, fname='lsst_ft_only_output_replace_smaller', unfrozen=['output'], long_title='LSST Fine Tuned Output')
    # experiment(pre=True, limits=limits, fname='lsst_ft_pref1', unfrozen=['gru_1', 'gru_2'], long_title='LSST Fine Tuned Pref')
    experiment(pre=True, limits=limits, fname='class_change_ft', unfrozen=['output'], long_title='LSST Class Change Fine Tuned')
    experiment(pre=False, limits=limits, fname='class_change_direct', unfrozen=None, long_title='LSST Class Change Direct Training')