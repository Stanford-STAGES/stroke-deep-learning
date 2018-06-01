import numpy as np
import h5py
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

dtu_red = [153/255, 0, 0]
dtu_grey = [153/255, 153/255, 153/255]
dtu_black = [0,0,0]
dtu_blue = [0, 0, 153/255]

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

path = '/home/rasmus/Desktop/shhs_subset/processed_data/shhs1-201199.hpf5'
with h5py.File(path, "r") as f:
    dset = np.array(f["x"])
    data = np.squeeze(dset[:, :, :])
    data = np.transpose(data, [1, 2, 0])

x = np.transpose(data,[2,1,0])
x = np.reshape(x, [2, -1])
decimation_factor = 1
fs = 125
fsd = fs / decimation_factor
n_channels, n_samples = x.shape
y = np.zeros([n_channels,n_samples//decimation_factor])
t = np.arange(0, y.shape[1]/fsd, 1/fsd)
for i in range(n_channels):
    y[i, :] = resample(x[i,:], n_samples=n_samples//decimation_factor)

lw = 0.6
font = {'size': 12}
channel_sep = 3.5
sns.set()
fig = plt.figure(figsize=(16,4))
whole_night = plt.subplot(3,1,1)
fivemin = plt.subplot(3,3,5)
fivesec = plt.subplot(3,5,13)

whole_night.plot(t,y[0,:]-channel_sep,'k', linewidth=lw)
whole_night.plot(t,y[1,:]+channel_sep,'k', linewidth=lw)
whole_night.set_xlim([0,8.7])
whole_night.set_xticks(np.arange(0.5*3600,8.6*3600,2*3600))
whole_night.set_xticklabels(['22:00:00', '00:00:00', '02:00:00', '04:00:00', '06:00:00'], fontdict=font)
whole_night.set_yticks((-channel_sep, channel_sep))
whole_night.set_yticklabels(['C4/A1', 'C3/A2'], fontdict=font)

fivemin_start = int(3.5*3600*fsd)
fivemin_end = int(fivemin_start + fsd*5*60)

facecolor = dtu_red
alpha = 0.5
edgecolor = dtu_red
low = np.min(y[0,:])-channel_sep
high = np.max(y[1,:])+channel_sep
rect = Rectangle([fivemin_start/fsd,low], 5*60, high-low)
pc = PatchCollection([rect], facecolor=facecolor, alpha=alpha,
                     edgecolor=edgecolor, zorder=1e3)
whole_night.add_collection(pc)

fivemin.plot(t[fivemin_start:fivemin_end], y[0,fivemin_start:fivemin_end]-channel_sep, color=dtu_red, linewidth=lw)
fivemin.plot(t[fivemin_start:fivemin_end], y[1,fivemin_start:fivemin_end]+channel_sep, color=dtu_red, linewidth=lw)
fivemin.set_yticks((-channel_sep, channel_sep))
fivemin.set_yticklabels(['C4/A1', 'C3/A2'], fontdict=font)
fivemin.set_xlim([fivemin_start/fsd, fivemin_end/fsd])
fivemin.set_xticks(np.arange(fivemin_start/fsd,fivemin_end/fsd+1,60))
fivemin.set_xticklabels(['22:00:00', '22:01:00', '22:02:00', '22:03:00', '22:04:00', '22:05:00'], fontdict=font)

fivesec_start = int(fivemin_start + 2*60*fsd)
fivesec_end = int(fivesec_start + fsd*5)

facecolor = dtu_grey
alpha = 0.5
edgecolor = dtu_grey
low = np.min(y[0,fivemin_start:fivemin_end])-channel_sep
high = np.max(y[1,fivemin_start:fivemin_end])+channel_sep
rect = Rectangle([fivesec_start/fsd,low], 5, high-low)
pc = PatchCollection([rect], facecolor=facecolor, alpha=alpha,
                     edgecolor=edgecolor, zorder=1e3)
fivemin.add_collection(pc)

fivesec.plot(t[fivesec_start:fivesec_end], y[0,fivesec_start:fivesec_end]-channel_sep, color=dtu_grey, linewidth=lw)
fivesec.plot(t[fivesec_start:fivesec_end], y[1,fivesec_start:fivesec_end]+channel_sep, color=dtu_grey, linewidth=lw)
fivesec.set_yticks((-channel_sep, channel_sep))
fivesec.set_yticklabels(['C4/A1', 'C3/A2'], fontdict=font)
fivesec.set_xlim([fivesec_start/fsd, fivesec_end/fsd])
fivesec.set_xticks(np.arange(fivesec_start/fsd,fivesec_end/fsd+1,5))
fivesec.set_xticklabels(['22:02:00', '22:02:05'], fontdict=font)

fig.patch.set_visible(False)
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
seg_length = int(5 * fsd)
lw = 0.4
time_shift = .6 #0.02
vertical_shift = 1
channel_sep = 5 #7
amp = 1
last_step = 13
n_to_show = 3
for step in range(60):
    if step == 0:
        T = t[seg_start:seg_end]
    step = 60-step

    #if not step in [0, 1, 2, 3, 4, 5, 6, 55, 56, 57, 58, 59]:
    #    continue
    #if not step in [0, 1, 2, 3, 4, 5, 6, 55, 56, 57, 58, 59, 60]:#[0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 25]:
    if step > last_step:
        continue
    if step in np.arange(n_to_show, last_step-n_to_show)+1:  # [0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 25]:
        axes[0, 0].plot(T[len(T)//2] + step * time_shift, vertical_shift * step,
                        '.k',
                        linewidth=2,
                        alpha=1,
                        zorder=60-step)#60-step)  # -step/60)
        continue

    seg_start = fivemin_start + step*seg_length
    seg_end = seg_start + seg_length
    axes[0, 0].plot(T+step*time_shift, amp*y[0,seg_start:seg_end]-channel_sep+vertical_shift*step,
                    color=dtu_grey, linewidth=lw,
                    alpha=1,
                    zorder=60-step)
    axes[0, 0].plot(T+step*time_shift, amp*y[1,seg_start:seg_end]+channel_sep+vertical_shift*step,
                    color=dtu_grey, linewidth=lw,
                    alpha=1,
                    zorder=60-step)

    facecolor = 'w'
    alpha = 1
    edgecolor = 'k'
    low = np.min(-5-channel_sep+vertical_shift*step)
    high = np.max(5+channel_sep+vertical_shift*step)

    if step == last_step:
        high_store = high
    rect = Rectangle([T[0]+step*time_shift, low], 5, high - low)
    pc = PatchCollection([rect], facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor, zorder=60-step, linewidth=1.5)

    axes[0, 0].add_collection(pc)

axes[0, 0].set_ylim([low, high_store*1.04])
axes[0, 0].axis('off')
fig.patch.set_visible(False)
plt.show()



###FRONTPAGE!
'''
fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
seg_length = int(5 * fsd)
lw = 0.4
time_shift = 0.1 #0.02
vertical_shift = 1
channel_sep = 7 #7
dtu_red = [154/255, 0, 0]
dtu_grey = [153/255, 153/255, 153/255]
amp = 1
for step in range(60):
    if step == 0:
        T = t[seg_start:seg_end]
    step = 60-step
    
    seg_start = fivemin_start + step*seg_length
    seg_end = seg_start + seg_length
    axes[0, 0].plot(T+step*time_shift, amp*y[0,seg_start:seg_end]-channel_sep+vertical_shift*step,
                    color=dtu_red, linewidth=lw,
                    alpha=1-step/60)
    axes[0, 0].plot(T+step*time_shift, amp*y[1,seg_start:seg_end]+channel_sep+vertical_shift*step,
                    color=dtu_grey, linewidth=lw,
                    alpha=1-step/60)
    
fig.patch.set_visible(False)
axes[0, 0].axis('off')
#axes[0, 0].set_xlim([t[seg_start], t[seg_end]])
#axes[0, 0].set_yticks([])
#axes[0, 0].set_xticks([])
plt.show()
'''