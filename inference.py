import argparse
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.contrib import predictor
from models import input_fn
from datahandler import DataHandler
from config import Config
import pickle

parser = argparse.ArgumentParser(description='Evaluate a stored model.')
parser.add_argument('--config', type=str, default='local',
                    help='name of storage configuration profile to use (defaults to local, defined in config.yaml)')
parser.add_argument('--experiment', type=str, default='basic',
                    help='name of experimental profile to use (defaults to basis, defined in experiment.yaml)')
parser.add_argument('--model', type=str, default='simple',
                    help='name of model/hyperparameter profile to use (defaults to default, defined in params.yaml)')
parser.add_argument('--hparam', type=str, default=None,
                    help='comma-seperated hparams and value, overrides model parameters from profile (defaults to None), format e.g.: --hparams=learning_rate=0.3.')
parser.add_argument('--cross_validate', type=int, default=None,
                    help='if True run inference one of 9 CV models')
args = parser.parse_args()

def serving_input_receiver_fn():
    '''
    Serving input receiver function for tf.Estimator.export_savedmodel
    :return: a TensorServingInputReceiver fitting the input data specified in cf.hparams
    '''
    features = tf.placeholder(shape=[None, cf.hparams.n_epoch_samples, cf.hparams.n_channels], dtype=tf.float32)
    receiver_tensors = features
    return tf.estimator.export.TensorServingInputReceiver(features, receiver_tensors)

if __name__ == "__main__":
    cf = Config(args.config,
                args.experiment,
                args.model,
                args.hparam).get_configs(cross_validate=args.cross_validate,
                                         matched=True)

    DataHandler.setup_partitions(cf,
                                 model_memory=True,
                                 cross_validate=args.cross_validate)

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=cf.eparams.save_checkpoint_steps,
                                        save_summary_steps=cf.eparams.save_summary_steps)

    model = Model('CRNN', cf.eparams)

    classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: model(features, labels, mode, cf.hparams),
        model_dir=cf.eparams.ckptdir,
        config=run_config)

    classifier.export_savedmodel(export_dir_base=cf.eparams.ckptdir,
                                 serving_input_receiver_fn=serving_input_receiver_fn)

    exports = [int(e) for e in os.listdir(cf.eparams.ckptdir) if e.isdigit()]
    export_dir = os.path.abspath(cf.eparams.ckptdir + str(exports[np.argmax(exports)]))
    predict_fn = predictor.from_saved_model(export_dir)

    valIDs = DataHandler.partitions['val']
    testIDs = DataHandler.partitions['test']
    matchedIDs = DataHandler.partitions['matched']

    val_probs = {}
    val_group = {}
    test_probs = {}
    test_group = {}
    matched_probs = {}
    matched_group = {}

    def run_inference(id, partition):
        features, labels = input_fn(partition, cf.eparams, id)
        prob = []
        while True:
            try:
                # this will break when input_fn can't make a full 16 times 5 min input
                # todo: use smaller batch, or zero pad for missing in batch
                x, y = sess.run([features, labels])
                predictions = predict_fn({"input": x})
                prob.append(np.transpose(predictions['probabilities'][:, 1]))
            except:
                print('{}: done processing {}'.format(partition, id))
                break
        return np.argmax(y[0,:]), np.reshape(np.asarray(prob), [-1])

    with tf.Session() as sess:
        for id in valIDs:
            val_group[id], val_probs[id] = run_inference(id, 'val_id')
        for id in testIDs:
            test_group[id], test_probs[id] = run_inference(id, 'test_id')
        for id in matchedIDs:
            matched_group[id], matched_probs[id] = run_inference(id, 'matched_id')

    with open(cf.eparams.ckptdir + 'eval/probabilities.pkl', 'wb') as f:
        pickle.dump([val_group, val_probs, test_group, test_probs, matched_probs, matched_group], f)