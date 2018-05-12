import argparse
import tensorflow as tf
import os
import numpy as np
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
args = parser.parse_args()

if __name__ == "__main__":
    cf = Config(args.config, args.experiment, args.model, None)
    DataHandler.setup_partitions(cf, model_memory = True)

    exports = [int(e) for e in os.listdir(cf.eparams.ckptdir) if e.isdigit()]
    export_dir = os.path.abspath(cf.eparams.ckptdir + str(exports[np.argmax(exports)]))
    predict_fn = predictor.from_saved_model(export_dir)

    valIDs = DataHandler.partitions['val']
    testIDs = DataHandler.partitions['test']
    val_probs = {}
    val_group = {}
    test_probs = {}
    test_group = {}
    with tf.Session() as sess:
        for id in valIDs:
            features, labels = input_fn('val_id', cf.eparams, id)
            prob = []
            while True:
                try:
                    x, y = sess.run([features, labels])
                    predictions = predict_fn({"input": x})
                    prob.append(np.transpose(predictions['probabilities'][:, 1]))
                except:
                    print('Done processing {}'.format(id))
                    break
            val_group[id] = np.argmax(y[0,:])
            val_probs[id] = np.reshape(np.asarray(prob), [-1])
        for id in testIDs:
            features, labels = input_fn('test_id', cf.eparams, id)
            prob = []
            while True:
                try:
                    x, y = sess.run([features, labels])
                    predictions = predict_fn({"input": x})
                    prob.append(np.transpose(predictions['probabilities'][:, 1]))
                except:
                    print('Done processing {}'.format(id))
                    break
            test_group[id] = np.argmax(y[0,:])
            test_probs[id] = np.reshape(np.asarray(prob), [-1])


    with open(cf.eparams.ckptdir + 'eval/probabilities.pkl', 'wb') as f:
        pickle.dump([test_group, test_probs, val_group, val_probs], f)

    #with open(cf.eparams.ckptdir + 'eval/probabilities.pkl', 'rb') as f:
    #    test_group, test_probs, val_group, val_probs = pickle.load(f)

