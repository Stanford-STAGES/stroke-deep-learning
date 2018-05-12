from tensorflow.contrib.training import HParams
from ruamel.yaml import YAML
from utils import determine_data_dimensions
from os.path import exists
from os import mkdir
import tensorflow as tf # Not unused even if grey - used in load of params.yaml

# Resource: https://hanxiao.github.io/2017/12/21/Use-HParams-and-YAML-to-Better-Manage-Hyperparameters-in-Tensorflow/

class YParams(HParams):
    def __init__(self, yaml_fn, config_name):
        super().__init__()
        with open(yaml_fn) as fp:
            for k, v in YAML().load(fp)[config_name].items():
                if isinstance(v, str) and v[0:2] == 'tf':
                    self.add_hparam(k, eval(v))
                else:
                    self.add_hparam(k, v)

class ModelParams(YParams):
    pass

class ExperimentParams(YParams):
    pass

class Config(YParams):
    def __init__(self, config_profile, experiment_profile, model_profile, overrides):
        # Setup basic configuration (storage)
        super().__init__('config.yaml', config_profile)

        # Setup experimental setup paramenters and name model based on profiles
        self.eparams = ExperimentParams('experiment.yaml', experiment_profile)
        self.eparams.add_hparam('ckptdir', self.logdir+config_profile+'_'+experiment_profile+'_'+model_profile+'/')

        # Setup model parameters/hyperparameters and determine data dimensions
        self.hparams = ModelParams('params.yaml', model_profile)
        if overrides:
            self.hparams.parse(overrides)
            overrides = overrides.replace('_','')
            overrides = overrides.replace('=','')
            self.eparams.ckptdir = self.eparams.ckptdir[0:-1] + '_' + overrides
            print(self.eparams.ckptdir)
        if not exists(self.logdir):
            print('Setting up logging directory at: {}.'.format(self.logdir))
            mkdir(self.logdir)
        (n_channels, n_epoch_samples, fs) = determine_data_dimensions(self.data_folder)
        self.hparams.add_hparam('n_channels', n_channels)
        self.hparams.add_hparam('n_epoch_samples', n_epoch_samples)
        self.hparams.add_hparam('n_sub_epoch_samples', n_epoch_samples // self.hparams.time_steps)
        self.hparams.add_hparam('fs', fs)

        self.add_hparam('n_channels', n_channels)
        self.add_hparam('n_epoch_samples', n_epoch_samples)
        self.add_hparam('fs', fs)
        #Override profile parameters if any are passed




