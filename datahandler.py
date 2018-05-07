from os import listdir
import h5py
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import sys
class DataHandler(object):

    @classmethod
    def setup_partitions(cls, data_folder, train_pct, val_pct, batch_sizes):
        # Get all IDs in data folder
        cls.data_folder = data_folder

        h5files = []
        for file in listdir(cls.data_folder):
            if file.endswith(".hpf5"):
                h5files.append(file)
        cls.ids = [f[:-5] for f in h5files]

        # Get data dimensions
        with h5py.File(cls.data_folder + cls.ids[0] + '.hpf5', "r") as f:
            (cls.n_channels, _, cls.n_epoch_samples) = f["x"].shape
            cls.fs = f['fs'][()]

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
                
        # Assign each ID to testing, training og validaiton partition
        y = [v for v in cls.labels.values()]
        ids = [k for k in cls.labels.keys()]
        testtrainsplit = StratifiedShuffleSplit(1,test_size=train_pct)
        testvalsplit = StratifiedShuffleSplit(1, test_size=val_pct)
        testtrain = next(testtrainsplit.split(np.zeros(len(y)), y))
        train = testtrain[1]
        test = testtrain[0]
        testval = next(testvalsplit.split(np.zeros(len(test)), [y[idx] for idx in test]))
        val = [test[idx] for idx in testval[0]]
        test = [test[idx] for idx in testval[1]]

        # Save determined partitions
        cls.partitions = {'train':[ids[idx] for idx in train],
                          'test': [ids[idx] for idx in test],
                          'val': [ids[idx] for idx in val]}

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

        # Store input batch_sizes
        cls.batch_sizes = batch_sizes
        pass

    @classmethod
    def get_partitions(cls):
        return cls.partitions
    @classmethod
    def set_partitions(cls, partitions):
        cls.partitions = partitions
        return cls.partitions
    @classmethod
    def get_fs(cls):
        return cls.fs
    @classmethod
    def get_partition(cls):
        return cls.partitions
    @classmethod
    def get_n_epochs(cls):
        return cls.n_epochs

    def __init__(self, subset):
        self.subset = subset
        self.partition = self.partitions[self.subset]
        self.batch_size = self.batch_sizes[subset]
        if subset == 'train' and self.batch_size != 16:
            print('Warning: batch-size different from 16 can give uneven batches due to exploration order.')
        self.n_classes = 2
        pass

    def __get_new_random_epoch(self, id):
        idx = self.current_epoch[id]
        if self.n_epochs[id]-1 == idx:
            if self.subset == 'train' or self.subset == 'val':
                print('Reshuffling for: ' + id)
                np.random.shuffle(DataHandler.epoch_order[id])
                DataHandler.current_epoch[id] = 0
                DataHandler.n_shuffles[id] += 1
                idx = 0
        DataHandler.current_epoch[id] += 1
        return self.epoch_order[id][idx]

    def __load_h5data_batch(self, id):

        with h5py.File(self.data_folder + id + '.hpf5', "r") as f:
            dset = f['x'] #np.array(f["x"])
            epoch = self.__get_new_random_epoch(id)
            data = np.squeeze(dset[:, epoch, :])
            data = np.transpose(data, [1, 0])

        return data, self.labels[id]

    def __load_h5data_sequence(self, id):
        with h5py.File(self.data_folder + id + '.hpf5', "r") as f:
            dset = np.array(f["x"])
            data = np.squeeze(dset[:, :, :])
            data = np.transpose(data, [1, 2, 0])
        return data, self.labels[id]

    def __get_exploration_order(self):
        idx = np.arange(len(self.partition))
        if self.subset == 'train':
                np.random.shuffle(idx)
        return idx

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
        '''
        try:
            while True:
                try:
                    idx = self.__get_exploration_order()
                    imax = len(idx) // self.batch_size
                    for i in range(imax):
                        batch = [self.partition[k] for k in idx[i * self.batch_size:(i + 1) * self.batch_size]]
                        X,y = self.__batch_generation(batch)
                        yield X,y
                except GeneratorExit:
                    print('Exit')
                    raise
                except Exception as e:
                    print(e)
                    sys.exit()
        finally:
            print('Closing...')
        '''
        while True:
            idx = self.__get_exploration_order()
            imax = len(idx)//self.batch_size
            for i in range(imax):
                batch = [self.partition[k] for k in idx[i*self.batch_size:(i+1)*self.batch_size]]
                X, y = self.__batch_generation(batch)
                #print(y)
                yield X, y


    def generate_sequence(self):
        batch_size = 16
        n = 1e6
        for i, id in enumerate(self.partition):
            n = np.min([self.n_epochs[id], n])
        imax = int(n) // batch_size
        for i, id in enumerate(self.partition):
            X, y = self.__sequence_generation(id)
            #imax = self.n_epochs[id] // batch_size
            for i in range(imax):
            #for i in range(1):
                Xb = X[i*batch_size:(i+1)*batch_size, :, :]
                yb = y[i*batch_size:(i+1)*batch_size]
                yield Xb, yb
