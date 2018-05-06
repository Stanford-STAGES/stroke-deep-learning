from os import listdir
import utils
import sys
import h5py

import numpy as np
import datetime
debugging = False
multimodal = False
simulated_data = True
rescale_mode = 'soft'

data_folder = '/home/rasmus/Desktop/shhs_subset/'
control = listdir(data_folder + 'control')
stroke = listdir(data_folder + 'stroke')

if debugging:
    control = control[0:1]
    stroke = stroke[0:1]

IDs = control+stroke
print(IDs)
group = np.concatenate((np.zeros(shape=len(control)),
                        np.ones(shape= len(stroke))))

if multimodal:
    channels_to_load = [2, 7] # todo: implement!!!
else:
    channels_to_load = [2, 7]

if multimodal:
    output_folder = '/home/rasmus/Desktop/shhs_subset/processed_data_multimodal/'
elif simulated_data:
    output_folder = '/home/rasmus/Desktop/shhs_subset/simulated_data/'
else:
    output_folder = '/home/rasmus/Desktop/shhs_subset/processed_data/'

for counter, ID in enumerate(IDs):
    print('Load, filtering, organizing: ' + str(ID) + ' (number ' + str(counter+1) + ' of ' + str(len(IDs)) + ').')
    if group[counter] == 1:
        filename = data_folder + 'stroke/' + ID
    else:
        filename = data_folder + 'control/' + ID
    data, filter = utils.load_edf_file(filename,
                                       channels_to_load,
                                       epoch_duration=5)

    if counter == 0:
        n_chans, n_epochs, n_epoch_samples = data["sigbufs"].shape
    output_file_name = output_folder +  ID[:-4] + ".hpf5"
    with h5py.File(output_file_name, "w") as f:
        x = data['sigbufs']
        if simulated_data:
            x = np.random.normal(0, 1, x.shape)
            if group[counter] == 1:
                x = utils.add_known_complex(x, data['fs'])
        x = utils.rescale(x, data['fs'], rescale_mode)
        dset = f.create_dataset("x", data=x, chunks=True)
        f['fs'] = data['fs']
        f["group"] = group[counter]

print("All files processed.")
now = datetime.datetime.now()

with open(output_folder + "conversion.log", "w") as log:
    log.write("Conversion for all files concluded on: " + str(now))
    log.write("\n Utilized channels: " + str(channels_to_load))
    log.write("\n Applied filter: " + filter["type"])
    log.write("\n\t   Low cut: " + str(filter["lowcut"]))
    log.write("\n\t   High cut: " + str(filter["highcut"]))
    log.write("\n Rejection of epochs based on sum of Pxx above 1000.)")
    log.write("\n Signal has been scaled by: " + str(rescale_mode))
    if simulated_data:
        log.write("\nSimulated data... All stroke subjects have had known element added.")

sys.exit()