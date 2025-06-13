from utils.load_bts_data import load_bts_data
from utils.load_ztf_data import load_ztf_data
from utils.load_lsst_data import load_lsst_data
# from utils.models import build_model, train, train_contextual
import numpy as np
import os
from utils.utils import save, load, get_auroc_cls, Results
from utils.models import transfer, build_model


from astromcad import astromcad

storage_path = "/ocean/projects/phy240020p/rgupta9/transfer_learning/"

os.makedirs(os.path.join(storage_path, 'bts'), exist_ok=True)



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
    # Load the BTS data
    X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, ordered_class_names, class_weights = load_bts_data(pad=656)
    bts = Dataset(X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, class_weights, ordered_class_names)
    print(bts.ordered_class_names)

    y_test_single = [np.argmax(bts.y_test[i]) for i in range(len(bts.y_test))]
    y_val_single = [np.argmax(bts.y_val[i]) for i in range(len(bts.y_val))]
    y_train_single = [np.argmax(bts.y_train[i]) for i in range(len(bts.y_train))]
    print("Testing Counts", np.unique(y_test_single, return_counts=True))
    print("Validation Counts", np.unique(y_val_single, return_counts=True))
    print("Training Counts", np.unique(y_train_single, return_counts=True))


    X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, ordered_class_names, class_weights, ntimesteps=load_ztf_data(True)
    ztf = Dataset(X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, class_weights, ordered_class_names)

    print(ztf.y_train.shape[-1])
    print(ztf.class_weights)
    # class_freq = None
    # X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, class_weights = load_lsst_data(656, sample_freq=class_freq)
    # lsst = Dataset(X_train, y_train, X_val, y_val, X_test, y_test, host_gal_train, host_gal_test, host_gal_val, class_weights)
    def experiment(pre, limits, fname, long_title, unfrozen):
        bts_direct_perf = {}
        
        for i in range(1, 6):
            if (pre):
                

                model = build_model(latent_size=100, ntimesteps=ztf.X_train.shape[1], num_classes=ztf.y_train.shape[1], contextual=0, n_features=ztf.X_train.shape[2])
                model_pth = os.path.join(storage_path, f"models/ztf_classifier_bts_new{i}.weights.h5") #_bts
                
                try:
                    model.load_weights(model_pth)
                except Exception as e:
                    print("Model could not load, defaulting to training model")
                    astromcad.train(model, ztf.X_train, ztf.y_train, ztf.X_val, ztf.y_val, ztf.class_weights, epochs=40)
                    # Save the model
                    model.save_weights(model_pth)
            
            for limit in limits:
                if (limit not in bts_direct_perf):
                    bts_direct_perf[limit] = {'class_based' : {cl : [] for cl in bts.ordered_class_names}, 'macro' : [], 'micro': []}

                new_model=None
                if (pre):
                    new_model = build_model(latent_size=100, ntimesteps=bts.X_train.shape[1], num_classes=bts.y_train.shape[1], contextual=0, n_features=bts.X_train.shape[2])
                    new_model.load_weights(model_pth)
                    # new_model = transfer(model_path=model_pth, ntimesteps=ztf.X_train.shape[1], n_features=ztf.X_train.shape[2], contextual=0, latent_size=100, old_output=ztf.y_train.shape[1], new_output=bts.y_train.shape[1])
                    for layer in new_model.layers:
                        if layer.name in unfrozen:
                            layer.trainable = True
                        else:
                            layer.trainable = False
                    print(new_model.summary())
                    
                else:
                    new_model = build_model(latent_size=100, ntimesteps=bts.X_train.shape[1], num_classes=bts.y_train.shape[1], contextual=0, n_features=bts.X_train.shape[2])
                
                val = limit//10
                train = limit - val
                astromcad.train(new_model, bts.X_train[:train], bts.y_train[:train], bts.X_val[:val], bts.y_val[:val], bts.class_weights, epochs=40)

                bts_pred = new_model.predict(bts.X_test)
                bts_true = [np.argmax(bts.y_test[i]) for i in range(len(bts.y_test))]

                performance = get_auroc_cls(bts_pred, bts_true)
                print(f"BTS Performance: {i} {limit} ", performance)


                bts_direct_perf[limit]['micro'].append(performance['micro'])
                bts_direct_perf[limit]['macro'].append(performance['macro'])

                by_class_performance = dict(zip(bts.ordered_class_names, performance['by_class']))
                for cl in bts.ordered_class_names:
                    bts_direct_perf[limit]['class_based'][cl].append(by_class_performance[cl])
                
                res = Results(bts_direct_perf, long_title)
                save(os.path.join(storage_path, f"bts/{fname}.pkl"), res)
        print(f"{long_title} Performance: ", bts_direct_perf)


    limits = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 5000, 6000]

    unfrozen = ['dense_3', 'latent', 'output']
    # experiment(pre=True, limits=limits, fname='bts_ft_last_two_five1', long_title='BTS Fine Tuned, Output and Latnet', unfrozen = ['output', 'latent'])
    experiment(pre=True, limits=limits, fname='bts_ft_output_three_new', long_title='BTS Fine Tuned Output', unfrozen=['output'])
    # experiment(pre=True, limits=limits, fname='bts_ft_all_five1', long_title='BTS Fine Tuned All', unfrozen=['gru1', 'gru2', 'dense_1', 'dense_3', 'latent', 'output'])
    experiment(pre=False, limits=limits, fname='bts_direct_three_new', long_title='BTS Direct Training', unfrozen=None)
