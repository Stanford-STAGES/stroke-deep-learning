from os import listdir
import h5py
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from os.path import exists
from shutil import rmtree
from os import mkdir
from numpy.random import shuffle
import sys

class DataHandler(object):
    @classmethod
    def setup_partitions(cls, config, model_memory, cross_validate = False):
        cls.n_channels = config.n_channels
        cls.n_epoch_samples = config.n_epoch_samples
        cls.fs = config.fs
        # Store batch_sizes in class
        cls.batch_sizes = config.eparams.batch_size
        # Get all IDs in data folder
        cls.data_folder = config.data_folder
        h5files = []
        for file in listdir(cls.data_folder):
            if file.endswith(".hpf5"):
                h5files.append(file)
        cls.ids = [f[:-5] for f in h5files]

        # Load each file and get its group and number of epochs
        cls.labels = {}
        cls.epoch_order = {}
        cls.n_epochs = {}
        cls.current_epoch = {}
        for i,id in enumerate(cls.ids):
            try:
                with h5py.File(cls.data_folder + id + '.hpf5', "r") as f:
                    cls.labels[id] = int(f["group"][()])
                    cls.n_epochs[id] = f['x'].shape[1]
                    cls.epoch_order[id] = np.arange(0,cls.n_epochs[id]-1)
                    cls.current_epoch[id] = 0
            except:
                print('Something is wrong with {}'.format(id))

        if cross_validate:
            #print('ERROR IN CV FOLD GENERATION, FIX. EXITING')
            #sys.exit()
            if model_memory and exists(config.eparams.ckptdir):
                print('Restoring partitions for: {}'.format(config.eparams.ckptdir))
                with open(config.eparams.ckptdir + 'partitions.pkl', 'rb') as f:
                    cls.partitions = pickle.load(f)
            else:
                n_folds = 9
                y = [v for v in cls.labels.values()]
                ids = [k for k in cls.labels.keys()]

                con = [k for k,v in cls.labels.items() if v == 0]
                exp = [k for k,v in cls.labels.items() if v == 1]
                shuffle(con)
                shuffle(exp)
                #n_con_per_split = len(con)//n_folds
                #n_exp_per_split = len(exp)//n_folds
                #n_con_extra = len(con) % n_folds
                #n_exp_extra = len(exp) % n_folds

                fold_idx_exp = np.tile(np.arange(0,n_folds), int(np.ceil(len(exp)/float(n_folds))))
                fold_idx_exp = fold_idx_exp[:len(exp)]
                fold_idx_con = np.tile(np.arange(0,n_folds), int(np.ceil(len(con)/float(n_folds))))
                fold_idx_con = fold_idx_con[:len(con)]

                fold_ids = []
                for i in range(n_folds):
                    con_ids = np.asarray(con)[fold_idx_con == i].tolist()
                    exp_ids = np.asarray(exp)[fold_idx_exp == i].tolist()
                    fold_ids.append(con_ids+exp_ids)

                #fold_ids = []
                #for i in range(n_folds):
                    # Value of boolean expression adds one for the "surplus" so all IDs are used, but the last splits are smaller

                #    con_ids = con[i*n_con_per_split + (i-1 < n_con_extra):i*n_con_per_split+n_con_per_split + (i < n_con_extra)]
                #    exp_ids = exp[i*n_exp_per_split + (i-1 < n_exp_extra):i*n_exp_per_split+n_exp_per_split + (i < n_exp_extra)]
                #    fold_ids.append(con_ids+exp_ids)


                division = []
                base = np.arange(0,n_folds).tolist() #[0,1,2,3,4,5,6,7,8,9]
                for n in range(n_folds):
                    division.append(base[n:] + base[:n])

                flatten = lambda l: [item for sublist in l for item in sublist]
                partitions_dicts = []
                for i in range(n_folds):
                    # Save determined partitions
                    idx = division[i]
                    partitions = {'train': flatten([fold_ids[e] for e in idx[0:n_folds-2]]),
                                      'test': fold_ids[idx[n_folds-2]],
                                      'val': fold_ids[idx[n_folds-1]]}
                    partitions_dicts.append(partitions)

                cls.partitions = partitions_dicts[cross_validate-1]

                base_dir = config.eparams.ckptdir[:-4]
                if exists(base_dir):
                    print('Removing existing model: {}'.format(base_dir))
                    rmtree(base_dir)
                mkdir(base_dir)
                with open(base_dir + 'cv_splits.pkl', 'wb') as f:
                    pickle.dump(partitions_dicts, f)
                for i in range(n_folds):
                    cv_fold_model_dir = base_dir+'/cv'+str(i+1)+'/'
                    mkdir(cv_fold_model_dir)
                    with open(cv_fold_model_dir + 'partitions.pkl', 'wb') as f:
                        pickle.dump(partitions_dicts[i], f)
        else:
            if model_memory and exists(config.eparams.ckptdir):
                print('Restoring partitions')
                with open(config.eparams.ckptdir + 'partitions.pkl', 'rb') as f:
                    cls.partitions = pickle.load(f)
            else:
                # Assign each ID to testing, training og validation partition
                y = [v for v in cls.labels.values()]
                ids = [k for k in cls.labels.keys()]
                ttsplit = StratifiedShuffleSplit(1, test_size=config.eparams.train_pct)
                tvsplit = StratifiedShuffleSplit(1, test_size=config.eparams.val_pct)
                tt = next(ttsplit.split(np.zeros(len(y)), y))
                train = tt[1]
                test = tt[0]
                tv = next(tvsplit.split(np.zeros(len(test)), [y[idx] for idx in test]))
                val = [test[idx] for idx in tv[0]]
                test = [test[idx] for idx in tv[1]]

                # Save determined partitions
                cls.partitions = {'train':[ids[idx] for idx in train],
                                  'test': [ids[idx] for idx in test],
                                  'val': [ids[idx] for idx in val]}

                if exists(config.eparams.ckptdir):
                    print('Removing existing model: {}'.format(config.eparams.ckptdir))
                    rmtree(config.eparams.ckptdir)

                mkdir(config.eparams.ckptdir)
                with open(config.eparams.ckptdir + 'partitions.pkl', 'wb') as f:
                    pickle.dump(cls.partitions, f)

        # Shuffle training and validation data and keep track of position
        cls.n_shuffles = {}
        for i, id in enumerate(cls.partitions['train']):
            np.random.shuffle(cls.epoch_order[id])
            cls.n_shuffles[id] = 1
        for i, id in enumerate(cls.partitions['val']):
            np.random.shuffle(cls.epoch_order[id])
            cls.n_shuffles[id] = 1
        for i, id in enumerate(cls.partitions['test']):
            cls.n_shuffles[id] = 1

        if config.matched_folder:
            # todo: validate
            cls.matched_folder = config.matched_folder
            h5files = []
            for file in listdir(cls.matched_folder):
                if file.endswith(".hpf5"):
                    h5files.append(file)
            cls.m_ids = [f[:-5] for f in h5files]

            for i, id in enumerate(cls.m_ids):
                try:
                    with h5py.File(cls.matched_folder + id + '.hpf5', "r") as f:
                        cls.labels[id] = int(f["group"][()])
                        cls.n_epochs[id] = f['x'].shape[1]
                        cls.epoch_order[id] = np.arange(0, cls.n_epochs[id] - 1)
                        cls.current_epoch[id] = 0
                except:
                    print('Something is wrong with {}'.format(id))
            cls.partitions['matched'] = cls.m_ids
            cls.batch_sizes['matched'] = 8

    @classmethod
    def set_partitions(cls, partitions):
        cls.partitions = partitions
        return cls.partitions

    #@classmethod
    #def get_fs(cls):
    #    return cls.fs

    #@classmethod
    #def get_partition(cls):
    #    return cls.partitions

    #@classmethod
    #def get_n_epochs(cls):
    #    return cls.n_epochs

    def __init__(self, subset, ID = None):
        self.subset = subset
        self.ID = ID
        self.partition = self.partitions[self.subset]
        self.batch_size = self.batch_sizes[subset]

        idx = np.arange(len(self.partition))
        ids = [self.partition[k] for k in idx]
        lab = np.asarray([int(self.labels[k]) for k in ids])
        self.exp = idx[lab == 1]
        self.con = idx[lab == 0]
        self.k = (self.batch_size//2)
        self.n_exp = len(self.exp)//self.k
        self.n_con = len(self.con)//self.k
        self.smallest = np.min([self.n_exp, self.n_con])

        self.n_classes = 2
        pass

    def __get_new_random_epoch(self, id):
        idx = self.current_epoch[id]
        if self.n_epochs[id]-1 == idx:
            if self.subset == 'train' or self.subset == 'val':
                #print('Reshuffling for: ' + id)
                np.random.shuffle(DataHandler.epoch_order[id])
                DataHandler.current_epoch[id] = 0
                DataHandler.n_shuffles[id] += 1
                idx = 0
        DataHandler.current_epoch[id] += 1
        return self.epoch_order[id][idx]

    def __load_h5data_batch(self, id):
        with h5py.File(self.data_folder + id + '.hpf5', "r") as f:
            dset = f['x']
            epoch = self.__get_new_random_epoch(id)
            data = np.squeeze(dset[:, epoch, :])
            data = np.transpose(data, [1, 0])
        return data, self.labels[id]

    def __load_h5data_sequence(self, id):
        if self.subset == 'matched':
            path = self.matched_folder + id + '.hpf5'
        else:
            path = self.data_folder + id + '.hpf5'

        with h5py.File(path, "r") as f:
                dset = np.array(f["x"])
                data = np.squeeze(dset[:, :, :])
                data = np.transpose(data, [1, 2, 0])

        return data, self.labels[id]

    def __get_exploration_order(self):
        k = self.k
        n = self.batch_size
        np.random.shuffle(self.exp)
        np.random.shuffle(self.con)
        idx = np.zeros(self.smallest*n)
        for i in range(self.smallest):
            c = np.concatenate([self.exp[i*k:i*k+k],self.con[i*k:i*k+k]])
            np.random.shuffle(c)
            idx[i*n:i*n+n] = c
        return idx.astype(int)

    def __sparsify(self, y):
        return np.array([[1 if y[i] == j else 0 for j in range(self.n_classes)]
                         for i in range(y.shape[0])])

    def __batch_generation(self, batch):
        X = np.empty((self.batch_size, self.n_epoch_samples, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, id in enumerate(batch):
            X[i, :, :], y[i] = self.__load_h5data_batch(id)
        return X, self.__sparsify(y)

    def __sequence_generation(self, id):
        X = np.empty((self.n_epochs[id], self.n_epoch_samples, self.n_channels))
        y = np.empty((self.n_epochs[id]), dtype=int)
        # Generate data
        X[:, :, :], y[:] = self.__load_h5data_sequence(id)
        return X, self.__sparsify(y)

    def generate_batches(self):
        while True:
            idx = self.__get_exploration_order()
            imax = len(idx)//self.batch_size
            for i in range(imax):
                batch = [self.partition[k] for k in idx[i*self.batch_size:(i+1)*self.batch_size]]
                X, y = self.__batch_generation(batch)
                yield X, y

    def generate_sequence(self):
        batch_size = 16
        n = 1e6
        for i, id in enumerate(self.partition):
            n = np.min([self.n_epochs[id], n])
        imax = int(n) // batch_size
        for i, id in enumerate(self.partition):
            X, y = self.__sequence_generation(id)
            for i in range(imax):
                Xb = X[i*batch_size:(i+1)*batch_size, :, :]
                yb = y[i*batch_size:(i+1)*batch_size]
                yield Xb, yb

    def generate_sequence_ID(self):
        batch_size = 16
        imax = int(self.n_epochs[self.ID]) // batch_size
        X, y = self.__sequence_generation(self.ID)
        for i in range(imax):
            Xb = X[i*batch_size:(i+1)*batch_size, :, :]
            yb = y[i*batch_size:(i+1)*batch_size]
            yield Xb, yb
