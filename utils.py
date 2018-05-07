import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.interpolate import interp1d
import pyedflib
import matplotlib.pyplot as plt
import itertools
import pandas as pd

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def load_edf_file_ssc(filename, channels_to_load):
    f = pyedflib.EdfReader(filename)
    labels = f.getSignalLabels()
    contained = {x: i for (i,x) in enumerate(labels) if x in channels_to_load.keys()}

    if not contained or len(contained) != 6:
        print(contained)
        return -1

    fss = f.getSampleFrequencies()
    fs = fss[contained['C3-A2']]
    n = f.getNSamples()[contained['C3-A2']]
    X = np.zeros((len(channels_to_load), n))
    lowcut = .3
    highcut = 40.
    for chan_name, chan_idx_in_file in contained.items():
        g = f.getPhysicalMaximum(chan_idx_in_file) / f.getDigitalMaximum(chan_idx_in_file)
        x = g*f.readSignal(chan_idx_in_file)
        if fss[chan_idx_in_file] != fs:
            time = np.arange(0, len(x) / fss[chan_idx_in_file], 1 / fs)
            t = np.arange(0, len(x) / fss[chan_idx_in_file], 1 / fss[chan_idx_in_file])
            F = interp1d(t, x, kind='linear', fill_value = 'extrapolate')
            x = F(time)
        X[channels_to_load[chan_name],:] = butter_bandpass_filter(x, lowcut, highcut, fs)

    #labels = [labels[x] for x in chan_number]
    data = {'x': X, 'fs': fs, 'labels': labels}
    return data

def load_edf_file(filename, channels_to_load, epoch_duration=5):
    # todo: gain correction?!
    f = pyedflib.EdfReader(filename)
    labels = f.getSignalLabels()
    #n_sigs = f.signals_in_file
    #channels_to_load = np.arange(0, n_sigs)  # [2,3,4,5,6,7]
    fss = f.getSampleFrequencies()
    n = f.getNSamples()[2]
    fs = fss[channels_to_load[0]]
    epoch_samples = int(epoch_duration * fs)
    n_epochs = n // epoch_samples
    sigbufs = np.zeros((len(channels_to_load), n_epochs, epoch_samples))

    time = np.arange(0, epoch_duration, 1 / fs)
    lowcut = .3
    highcut = 40.
    for i, j in enumerate(channels_to_load):
        x = f.readSignal(j)
        if fss[j] != fs:
            #x = resample(x, n, np.arange(0, len(x) / fss[j], 1 / fss[j]))
            t = np.arange(0, len(x) / fss[j], 1 / fss[j])
            F = interp1d(t,x,kind='linear')
            x = F(time)
        if i in channels_to_load:
            x = butter_bandpass_filter(x, lowcut, highcut, fs)

        sigbufs[i, :, :] = np.asarray(list(zip(*[iter(x)] * epoch_samples)))

        #vals = []
        #accept = np.zeros((n_epochs))
        reject = []
        for j in range(0,n_epochs):
            fxx, Pxx = welch(sigbufs[i,j,:], fs=fs)
            s = np.sum(Pxx)
            if s > 2000: #s > 1
                reject.append(j)
        #print(accept==1)
        #tmp = sigbufs[:, accept == 1, :]
        #sigbufs = tmp
        #vals.append(np.sum(Pxx))
        #print(reject)
        #import matplotlib.pyplot as plt
        #x = np.asarray(vals)
        #n, bins, patches = plt.hist(x, bins=100, facecolor='blue', alpha=0.5)
        #plt.title(str(np.median(vals)))
        #plt.show()
        #sys.exit()
    #print(len(reject))
    if len(reject) != 0:
        reject = np.unique(np.asarray(reject))
        mask = np.ones((n_epochs), dtype=bool)
        mask[reject] = False
        tmp = sigbufs[:, mask, :]
        sigbufs = tmp
    filter = {"type": "butter_bandpass", "lowcut": lowcut, "highcut": highcut}
    labels = [labels[x] for x in channels_to_load]
    data = {'sigbufs': sigbufs, 'fs': fs, 'labels': labels}
    return data, filter

def load_hypnogram_file(filename):
    with open(filename, 'r') as f:
        reader = pd.read_csv(f, delimiter='   ', header = None, names = ['t','stage'] )
        hyp = reader.as_matrix(['stage'])
        hyp = np.squeeze(hyp)
        hyp = hyp.astype(int)
    return hyp

def visualize_epoch(data, epoch = 1):
    '''use with load_edf_file:
       data = utils.load_edf_file(filename, channels_to_load)
       utils.visualize_epoch(data, epoch = 300)
       '''
    time = np.arange(0, 30, 1 / data['fs'])
    X = data['sigbufs']
    labels = data['labels']
    (n_chans, n_epochs, n_samples) = X.shape
    fig, axs = plt.subplots(nrows=n_chans, figsize=(6, 10))
    for i, j in enumerate(axs):
        j.plot(time, X[i, epoch, :])
        j.set_title(labels[i])
        j.autoscale(enable=True, axis='x', tight=True)
    plt.show()

from sklearn.preprocessing import StandardScaler
def rescale(x, fs, mode, window_duration = 5):
    if mode == 'running':
        mean = np.zeros(x.shape)
        M = np.round(fs*window_duration)

        for chan in [0, 1]:
            cumsum = np.cumsum(np.insert(x[:,chan], 0, 0))
            mean[:-(M-1),chan] =  (cumsum[M:] - cumsum[:-M]) / float(M)

        var = ((x - mean)**2)

        for chan in [0, 1]:
            cumsum = np.cumsum(np.insert(var[:,chan], 0, 0))
            var[:-(M-1),chan] =  (cumsum[M:] - cumsum[:-M]) / float(M)

        new_x = (x - mean) / (np.sqrt(var))
    elif mode == 'standardscaler':
        scaler = StandardScaler()
        new_x = scaler.fit_transform(x)
    elif mode == 'soft':
        q5 = np.percentile(x,5)
        q95 = np.percentile(x,95)
        new_x = 2*(x-q5)/(q95-q5) - 1
    return new_x

from scipy.signal import gaussian, sawtooth
from copy import deepcopy
from numpy.random import randint as randi
from numpy.random import normal as randn
def add_known_complex(x, fs, easy = False, window_duration = 3):
    if easy:
        M = np.round(fs * window_duration)
        g = gaussian(M, std=50)
        t = np.arange(0,g.shape[0],1)/fs
        z = np.sin(2*3.14*10*t)*2
        w = g*z
        for i in range(0, x.shape[1]):
            x[0, i, 100:100 + M] = x[0, i, 100:100 + M] + w
    else:
        for i in range(0, x.shape[1]):
            # Sine complex
            M = np.round(fs * window_duration)
            M = M + randi(-M,M)//2
            g = gaussian(M, std=50)
            t = np.arange(0, g.shape[0], 1) / fs
            amplitude = randn(0,.5)+8
            frequency = 10+randn(0,2)
            z = np.sin(2 * np.pi * frequency * t) * amplitude
            w = g * z
            chan = randi(0,2)
            start_index = randi(0, x.shape[2]-M)
            x[chan, i, start_index:start_index + M] = x[chan, i, start_index:start_index + M] + w

            # Saw-tooth complex
            M = np.round(fs * window_duration)
            M = M + randi(-M, M) // 2
            g = gaussian(M, std=100)
            t = np.arange(0, g.shape[0], 1) / fs
            amplitude = randn(0, .5) + 6
            frequency = 2 + randn(0, .1)
            z = sawtooth(2 * np.pi * frequency * t) * amplitude
            w = g * z
            chan = randi(0, 2)
            start_index = randi(0, x.shape[2] - M)
            x[chan, i, start_index:start_index + M] = x[chan, i, start_index:start_index + M] + w

    #new_x = deepcopy(x)
    #for i in range(0,x.shape[1]):
    #    new_x[0, i, 100:100+M] = new_x[0, i, 100:100+M] + w
    #fig, ax = plt.subplots(ncols=1, nrows=2)
    #ax[0].plot(x[0,0,:])
    #ax[1].plot(new_x[0,0,:])
    #plt.show()
    return x

def determine_data_dimensions(data_folder):
    from os import listdir
    import h5py
    h5files = []
    for file in listdir(data_folder):
        if file.endswith(".hpf5"):
            h5files.append(file)
    ids = [f[:-5] for f in h5files]
    # Get data dimensions
    with h5py.File(data_folder + ids[0] + '.hpf5', "r") as f:
        (n_channels, _, n_epoch_samples) = f["x"].shape
        fs = f['fs'][()]
    return (n_channels, n_epoch_samples, fs)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', vmin=0, vmax=1,cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
