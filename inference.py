import argparse
import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib import predictor
from models import input_fn
from models import CRNN as Model
from datahandler import DataHandler
from config import Config
import pickle
import traceback

parser = argparse.ArgumentParser(description='Evaluate a stored model.')
parser.add_argument('--config', type=str, default='shhs',
                    help='name of storage configuration profile to use (defaults to local, defined in config.yaml)')
parser.add_argument('--experiment', type=str, default='local',
                    help='name of experimental profile to use (defaults to basis, defined in experiment.yaml)')
parser.add_argument('--model', type=str, default='small',
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

    trainIDs = DataHandler.partitions['train']
    valIDs = DataHandler.partitions['val']
    testIDs = DataHandler.partitions['test']
    matchedIDs = DataHandler.partitions['matched']
    do_td = False
    train_probs = {}
    train_group = {}
    train_feat = {}
    #train_sens = {}
    val_probs = {}
    val_group = {}
    val_feat = {}
    #val_sens = {}
    test_probs = {}
    test_group = {}
    test_feat = {}
    test_sens = {}
    matched_probs = {}
    matched_group = {}
    matched_feat = {}
    matched_sens = {}
    total_records = len(trainIDs + valIDs + testIDs + matchedIDs)

    def run_inference(id, partition, td = False):
        features, labels = input_fn(partition, cf.eparams, id)
        prob = []
        feat = []
        exp_sens = []
        con_sens = []
        while True:
            try:
                # this will break when input_fn can't make a full 16 times 5 min input
                # todo: use smaller batch, or zero pad for missing in batch
                x, y = sess.run([features, labels])
                predictions = predict_fn({"input": x})
                prob.append(np.transpose(predictions['probabilities'][:, 1]))
                feat.append(np.transpose(predictions['features']))
                if td:
                    exp_sens.append(predictions['experimental_sensitivity'])
                    con_sens.append(predictions['control_sensitivity'])
            except:
                #print('{}: done processing {}'.format(partition, id))
                break

        features = np.reshape(np.transpose(np.asarray(feat), [0, 2, 1]), [len(feat)*4, -1])
        if not td:
            return np.argmax(y[0,:]), np.reshape(np.asarray(prob), [-1]), features
        else:
            td_exp = np.reshape( np.transpose(np.asarray(exp_sens), [2, 4, 5, 0, 1, 3]), [2, 625, -1])
            td_con = np.reshape( np.transpose(np.asarray(con_sens), [2, 4, 5, 0, 1, 3]), [2, 625, -1])
            taylor_decomp = np.stack([td_exp, td_con], axis=3)
            return np.argmax(y[0,:]), np.reshape(np.asarray(prob), [-1]), features, taylor_decomp

    with tf.Session() as sess:
        for step, id in enumerate(trainIDs):
            try:
                train_group[id], train_probs[id], train_feat[id] = run_inference(id, 'train_id')
                print('{}: done processing {} ({} of {})'.format('train_id', id, step+1, len(trainIDs)))
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print('{}: error processing {}'.format('train_id', id))
        print('Done processing {} of a total of {}'.format(len(trainIDs), total_records))
        for step,id in enumerate(valIDs):
            try:
                val_group[id], val_probs[id], val_feat[id] = run_inference(id, 'val_id')
                print('{}: done processing {} ({} of {})'.format('val_id', id, step+1, len(valIDs)))
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print('{}: error processing {}'.format('val_id', id))
        print('Done processing {} of a total of {}'.format(len(trainIDs+valIDs), total_records))

        for step,id in enumerate(testIDs):
            try:
                if do_td:
                    test_group[id], test_probs[id], test_feat[id], test_sens[id] = run_inference(id, 'test_id', td=do_td)
                else:
                    test_group[id], test_probs[id], test_feat[id] = run_inference(id, 'test_id')
                print('{}: done processing {} ({} of {})'.format('test_id', id, step+1, len(testIDs)))
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print('{}: error processing {}'.format('test_id', id))
        print('Done processing {} of a total of {}'.format(len(trainIDs+valIDs+testIDs), total_records))

        for step,id in enumerate(matchedIDs):
            try:
                if do_td:
                    matched_group[id], matched_probs[id], matched_feat[id], matched_sens[id] = run_inference(id, 'matched_id', td=do_td)
                else:
                    matched_group[id], matched_probs[id], matched_feat[id] = run_inference(id, 'matched_id')
                print('{}: done processing {} ({} of {})'.format('matched_id', id, step+1, len(matchedIDs)))
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print('{}: error processing {}'.format('matched_id', id))

    print('Exporting probabilities to: {}'.format(cf.eparams.ckptdir + 'eval/probabilities.pkl'))
    with open(cf.eparams.ckptdir + 'eval/probabilities.pkl', 'wb') as f:
        pickle.dump([train_group, train_probs, val_group, val_probs, test_group, test_probs, matched_probs, matched_group], f)

    print('Exporting features to: {}'.format(cf.eparams.ckptdir + 'eval/features.pkl'))
    with open(cf.eparams.ckptdir + 'eval/features.pkl', 'wb') as f:
        pickle.dump(
            [train_group, train_feat, val_group, val_feat, test_group, test_feat, matched_feat, matched_group],
            f)

    if do_td:
        print('Exporting taylor decompositions to: {}'.format(cf.eparams.ckptdir + 'eval/taylor_decompositions.pkl'))
        with open(cf.eparams.ckptdir + 'eval/taylor_decompositions.pkl', 'wb') as f:
            pickle.dump(
                [test_sens, test_group, matched_sens, matched_group], f)

    print('Done processing all.')
