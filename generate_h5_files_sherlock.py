from os import listdir
import utils
import sys
import h5py
import numpy as np
import pandas as pd
import datetime

rescale_mode = 'soft'
data_folder = '/scratch/PI/mignot/nsrr/shhs/polysomnography/edfs/shhs1/'
df = pd.read_csv('/home/users/rmth/stroke-deep-learning/IDs.csv')
IDs = np.asarray(df['IDs'])
group = np.asarray(df['group'])
channels_to_load = [2, 7]
#output_folder = '/home/users/rmth/processed_shhs_data/'
output_folder = '/scratch/users/rmth/processed_shhs_data/'

for counter, ID in enumerate(IDs):
    try:
    	print('Load, filtering, organizing: ' + str(ID) + ' (number ' + str(counter+1) + ' of ' + str(len(IDs)) + ').')
    
    	filename = data_folder + 'shhs1-' + str(ID) + '.edf'
    	print(filename)
    	data, filter = utils.load_edf_file(filename,
                                       channels_to_load,
                                       epoch_duration=5)
    	if counter == 0:
        	n_chans, n_epochs, n_epoch_samples = data["sigbufs"].shape
    	output_file_name = output_folder + 'shhs1-' + str(ID) + ".hpf5"
    	with h5py.File(output_file_name, "w") as f:
        	x = data['sigbufs']
        	x = utils.rescale(x, data['fs'], rescale_mode)
        	dset = f.create_dataset("x", data=x, chunks=True)
        	f['fs'] = data['fs']
        	f['group'] = int(group[counter])
    except: 
	print('Error happend while processing: {}'.format(str(filename))

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
