from tensorflow.contrib.training import HParams
from ruamel.yaml import YAML
from utils import determine_data_dimensions
from os.path import exists
from os import mkdir
import tensorflow as tf # Not unused even if grey - used in load of params.yaml

# Resource: https://hanxiao.github.io/2017/12/21/Use-HParams-and-YAML-to-Better-Manage-Hyperparameters-in-Tensorflow/

def new_hparam(yaml_fn, config_name):
    hparam = HParams()
    with open(yaml_fn) as fp:
        for k, v in YAML().load(fp)[config_name].items():
            if isinstance(v, str) and v[0:2] == 'tf':
                hparam.add_hparam(k, eval(v))
            else:
                hparam.add_hparam(k, v)
    return hparam

class Config():
    def __init__(self, config_profile, experiment_profile, model_profile, overrides):
        self.c = config_profile
        self.e = experiment_profile
        self.m = model_profile
        self.o = overrides

    def get_configs(self, matched = False):
        hparam = new_hparam('/home/users/rmth/stroke-deep-learning/config.yaml', self.c)
        if matched:
            hparam.add_hparam('matched_folder', hparam.data_folder + 'matched_controls/')
        else:
            hparam.add_hparam('matched_folder', None)

        # Setup experimental setup paramenters and name model based on profiles
        hparam.eparams = new_hparam('/home/users/rmth/stroke-deep-learning/experiment.yaml', self.e)
        hparam.eparams.add_hparam('ckptdir', hparam.logdir+self.c+'_'+self.e+'_'+self.m+'/')

        # Setup model parameters/hyperparameters and dete rmine data dimensions
        hparam.hparams = new_hparam('/home/users/rmth/stroke-deep-learning/params.yaml', self.m)
        if self.o:
            hparam.hparams.parse(self.o)
            self.o = self.o.replace('_', '')
            self.o = self.o.replace('=', '')
            hparam.eparams.ckptdir = hparam.eparams.ckptdir[0:-1] + '_' + self.o + '/'
            #print(self.eparams.ckptdir)
        if not exists(hparam.logdir):
            print('Setting up logging directory at: {}.'.format(hparam.logdir))
            mkdir(hparam.logdir)
        (n_channels, n_epoch_samples, fs) = determine_data_dimensions(hparam.data_folder)
        hparam.hparams.add_hparam('n_channels', n_channels)
        hparam.hparams.add_hparam('n_epoch_samples', n_epoch_samples)
        hparam.hparams.add_hparam('n_sub_epoch_samples', n_epoch_samples // hparam.hparams.time_steps)
        hparam.hparams.add_hparam('fs', fs)

        hparam.add_hparam('n_channels', n_channels)
        hparam.add_hparam('n_epoch_samples', n_epoch_samples)
        hparam.add_hparam('fs', fs)
        return hparam
        #Override profile parameters if any are passed




