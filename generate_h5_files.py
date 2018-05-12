from os import listdir
import utils
import h5py
import numpy as np
import pandas as pd

# todo: implement rejection of noise epochs
debugging = False
multimodal = True
simulated_data = False
rescale_mode = 'soft'
cohort = 'SHHS'

if cohort == 'SSC':
    edf_folder = '/home/rasmus/Desktop/SSC/raw/edf/'
    hypnogram_folder = '/home/rasmus/Desktop/SSC/raw/hypnograms/'
    epoch_duration = 5
    channels_to_load = {'C3': 0, 'C4': 1, 'Fz': 2, 'ROC': 3, 'LOC': 4, 'Chin': 5}
    output_folder = '/home/rasmus/Desktop/SSC/processed_data/'
    IDs = listdir(edf_folder)
    channel_alias = utils.read_channel_alias(edf_folder+'signal_labels.json')
elif cohort == 'SHHS-Sherlock':
    epoch_duration = 5*60
    edf_folder = '/scratch/PI/mignot/nsrr/shhs/polysomnography/edfs/shhs1/'
    hypnogram_folder = None
    df = pd.read_csv('/home/users/rmth/stroke-deep-learning/IDs.csv')
    IDs = np.asarray(df['IDs'])
    group = np.asarray(df['group'])
    if multimodal:
        channels_to_load = {'eeg1': 0, 'eeg2': 1, 'ecg': 2, 'pulse': 3}
        output_folder = '/scratch/users/rmth/processed_data_multimodal/'
        channel_alias = utils.read_channel_alias(edf_folder+'signal_labels_multimodal.json')
    else:
        channels_to_load = {'eeg1': 0, 'eeg2': 1}
        output_folder = '/scratch/users/rmth/processed_shhs_data/'
        channel_aliases = None
elif cohort == 'SHHS':
    epoch_duration = 5*60
    edf_folder = '/home/rasmus/Desktop/shhs_subset/'
    hypnogram_folder = None
    control = listdir(edf_folder + 'control')
    stroke = listdir(edf_folder + 'stroke')
    if debugging:
        control = control[0:1]
        stroke = stroke[0:1]
    group = np.concatenate((np.zeros(shape=len(control)),
                            np.ones(shape= len(stroke))))
    IDs = control+stroke
    if multimodal:
        channels_to_load = {'eeg1': 0, 'eeg2': 1, 'ecg': 2, 'pulse': 3}
        output_folder = '/home/rasmus/Desktop/shhs_subset/processed_data_multimodal/'
        channel_alias = utils.read_channel_alias(edf_folder+'signal_labels_multimodal.json')
    elif simulated_data:
        output_folder = '/home/rasmus/Desktop/shhs_subset/simulated_data/'
    else:
        channels_to_load = {'eeg1': 0, 'eeg2': 1}
        output_folder = '/home/rasmus/Desktop/shhs_subset/processed_data/'
        channel_alias = utils.read_channel_alias(edf_folder+'signal_labels.json')

for counter, ID in enumerate(IDs):
    try:
        print('Processing: ' + str(ID) + ' (number ' + str(counter+1) + ' of ' + str(len(IDs)) + ').')
        if cohort == 'SSC':
            filename = edf_folder + ID
        elif cohort == 'SHHS-Sherlock':
            filename = edf_folder + 'shhs1-' + str(ID) + '.edf'
        elif cohort == 'SHHS':
            if group[counter] == 1:
                filename = edf_folder + 'stroke/' + ID
            else:
                filename = edf_folder + 'control/' + ID

        try:
            data = utils.load_edf_file(filename, channels_to_load,
                                       cohort = cohort,
                                       channel_alias = channel_alias)
        except:
            print('    EDF error (loading failed).')
            continue
        if data == -1:
            print('    Ignoring this subject due to different header setup.')
            continue

        x = data['x']
        n = x.shape[1]
        epoch_samples = epoch_duration * data['fs']
        n_epochs = n // epoch_samples

        if hypnogram_folder:
            filename = hypnogram_folder + ID[:-4] + '.STA'
            try:
                hypnogram = utils.load_hypnogram_file(filename)
                if epoch_duration != 30:
                    hypnogram = np.repeat(hypnogram, repeats=30 // epoch_duration)
                if n != hypnogram.shape[0]:
                    hypnogram = hypnogram[:n_epochs]
            except:
                print('    Hypnogram error')
                continue

        epoched = np.zeros((len(channels_to_load), n_epochs, epoch_samples))
        for i in range(len(channels_to_load)):
            epoched[i, :, :] = np.asarray(list(zip(*[iter(x[i,:])] * epoch_samples)))

        if simulated_data:
            x = np.random.normal(0, 1, x.shape)
            if group[counter] == 1:
                x = utils.add_known_complex(x, data['fs'])

        x = utils.rescale(epoched, data['fs'], rescale_mode)

        if cohort == 'SSC':
            stages_to_use = [5,2]
            stage_names = ['wake','n1','n2','n3','n4','rem','unknown','artefact']
            for group, stage in enumerate(stages_to_use):
                output_file_name = output_folder +  ID[:-4] + '_' + stage_names[stage] + ".hpf5"
                with h5py.File(output_file_name, "w") as f:
                    dset = f.create_dataset("x", data=x[:, hypnogram == stage, :], chunks=True)
                    f['fs'] = data['fs']
                    f["group"] = group
        elif cohort == 'SHHS-Sherlock':
            output_file_name = output_folder + 'shhs1-' + str(ID) + ".hpf5"
            with h5py.File(output_file_name, "w") as f:
                dset = f.create_dataset('x', data=x, chunks=True)
                f['fs'] = data['fs']
                f['group'] = int(group[counter])
        elif cohort == 'SHHS':
            output_file_name = output_folder +  ID[:-4] + ".hpf5"
            with h5py.File(output_file_name, "w") as f:
                dset = f.create_dataset("x", data=x, chunks=True)
                f['fs'] = data['fs']
                f["group"] = group[counter]
    except:
        print('Error happened while processing: {}'.format(str(filename)))

print("All files processed.")
