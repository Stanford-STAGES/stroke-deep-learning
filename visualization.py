import numpy as np
import h5py
from sklearn.utils import resample
import matplotlib.pyplot as plt

path = '/home/rasmus/Desktop/shhs_subset/processed_data/shhs1-201199.hpf5'
with h5py.File(path, "r") as f:
    dset = np.array(f["x"])
    data = np.squeeze(dset[:, :, :])
    data = np.transpose(data, [1, 2, 0])

x = np.transpose(data,[2,1,0])
x = np.reshape(x, [2, -1])
decimation_factor = 10
fs = 125
n_channels, n_samples = x.shape
y = np.zeros([n_channels,n_samples//decimation_factor])
t = np.arange(0, (n_samples//decimation_factor)/fs, 1/fs)
for i in range(n_channels):
    y[i, :] = resample(x[i,:], n_samples=n_samples//decimation_factor)


g = 1 / (1 + np.exp(-1e-1*(t-t[-1]//2)))
plt.plot(t,g)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)

ax[0,0].plot(g,y[0,:]-5)
ax[0,0].plot(g,y[1,:]+5)
plt.xticks(g, str(t.tolist()))

plt.show()