from os import listdir
import utils
import h5py
import numpy as np

rescale_mode = 'soft'
edf_folder = '/home/rasmus/Desktop/SSC/raw/edf/'
hypnogram_folder = '/home/rasmus/Desktop/SSC/raw/hypnograms/'
IDs = listdir(edf_folder)
# Add EOG, and a FRONTAL
#channels_to_load = ['C3-A2','C4-A1', 'Fz-A2', 'ROC-A1', 'LOC-A2', 'Chin EMG']
channels_to_load = {'C3-A2': 0, 'C4-A1': 1, 'Fz-A2': 2, 'ROC-A1': 3, 'LOC-A2': 4, 'Chin EMG': 5}
output_folder = '/home/rasmus/Desktop/SSC/processed_data/'

for counter, ID in enumerate(IDs):
    print('Processing: ' + str(ID) + ' (number ' + str(counter+1) + ' of ' + str(len(IDs)) + ').')
    filename = edf_folder + ID

    #if ID != 'SSC_1558_1.EDF':
    #    continue

    try:
        data = utils.load_edf_file_ssc(filename,
                                           channels_to_load)
    except:
        print('edf error')
        continue

    if data == -1:
        print('Ignoring this subject due to different header setup.')
        continue

    filename = hypnogram_folder + ID[:-4] + '.STA'
    try:
        hypnogram = utils.load_hypnogram_file(filename)
    except:
        print('hyp error')
        continue

    x = data['x']
    epoch_duration = 5
    n = x.shape[1]
    epoch_samples = epoch_duration * data['fs']
    n_epochs = n // epoch_samples
    if epoch_duration != 30:
        hypnogram = np.repeat(hypnogram, repeats = 30//epoch_duration)

    sigbufs = np.zeros((len(channels_to_load), n_epochs, epoch_samples))
    for i in range(len(channels_to_load)):
        sigbufs[i, :, :] = np.asarray(list(zip(*[iter(x[i,:])] * epoch_samples)))
    x = utils.rescale(sigbufs, data['fs'], rescale_mode)
    # todo: check if this is wrong:
    if x.shape[1] != hypnogram.shape[0]:
        hypnogram = hypnogram[:x.shape[1]]

    stages_to_use = [5,2]
    stage_names = ['wake','n1','n2','n3','n4','rem','unknown','artefact']
    for group, stage in enumerate(stages_to_use):
        output_file_name = output_folder +  ID[:-4] + '_' + stage_names[stage] + ".hpf5"
        with h5py.File(output_file_name, "w") as f:
            dset = f.create_dataset("x", data=x[:, hypnogram == stage, :], chunks=True)
            print(dset.shape)
            #f['x'] = x[:, hypnogram == stage, :]
            f['fs'] = data['fs']
            f["group"] = group

print("All files processed.")