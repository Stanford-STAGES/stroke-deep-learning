import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.interpolate import interp1d
import pyedflib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import gaussian, sawtooth
from numpy.random import randint as randi
from numpy.random import normal as randn
import json

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

def load_edf_file(filename, channels_to_load, cohort, channel_alias):
    f = pyedflib.EdfReader(filename)
    labels = f.getSignalLabels()
    contained = {channel_alias[e]: i for (i, e) in enumerate(labels) if e in channel_alias}
    if not contained or len(contained) != len(channels_to_load):
        print(labels)
        print(contained)
        return -1

    fss = f.getSampleFrequencies()
    if cohort == 'SSC':
        fs = fss[contained['C3']]
        n = f.getNSamples()[contained['C3']]
    elif cohort == 'SHHS' or cohort == 'SHHS-Sherlock' or cohort=='SHHS-Sherlock-matched':
        fs = fss[contained['eeg1']]
        n = f.getNSamples()[contained['eeg2']]
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
    data = {'x': X, 'fs': fs, 'labels': labels}
    return data

def reject_noise_epochs(sigbufs, fs):
    # todo: implement properly
    print('Implement utils.reject_noise_epochs...')
    return -1
    i = 1
    n_epochs = 1
    reject = []
    for j in range(0,n_epochs):
        fxx, Pxx = welch(sigbufs[i,j,:], fs=fs)
        s = np.sum(Pxx)
        if s > 2000:
            reject.append(j)
    if len(reject) != 0:
        reject = np.unique(np.asarray(reject))
        mask = np.ones((n_epochs), dtype=bool)
        mask[reject] = False
        tmp = sigbufs[:, mask, :]
        sigbufs = tmp
    return sigbufs

def load_hypnogram_file(filename):
    with open(filename, 'r') as f:
        reader = pd.read_csv(f, delimiter='   ', header = None, names = ['t','stage'] )
        hyp = reader.as_matrix(['stage'])
        hyp = np.squeeze(hyp)
        hyp = hyp.astype(int)
    return hyp

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

def read_channel_alias(fp):
    # Based on output from Hyatt's channel_label_identifier
    with open(fp) as f:
        data = json.load(f)
    alias = {}
    for chan in data['categories']:
        for a in data[chan]:
            alias[a] = chan
    return alias
