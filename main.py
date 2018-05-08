import os
import pickle
from sys import exit as q

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from getpass import getuser
from datahandler import DataHandler
from models import SimpleCRNN as Model
from models import input_fn
from utils import plot_confusion_matrix, determine_data_dimensions
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

verbose = True
simulated_data = False
n2rem_data = False  # control: rem, experimental: n2
model_memory = False
train_model = True
evaluate_model = False
export_savedmodel = False
activation_maximization = False
std_peak_analysis = False
visualize_std = False
gan_activation_maximization = False
user = getuser()
if user == 'rasmus':
    logdir = './tf_logs/'
    ckptdir = logdir + 'model/'
    # Define folder where data is located
    if simulated_data:
        data_folder = '/home/rasmus/Desktop/shhs_subset/simulated_data/'
    elif n2rem_data:
        data_folder = '/home/rasmus/Desktop/SSC/processed_data/'
    else:
        data_folder = '/home/rasmus/Desktop/shhs_subset/processed_data/'
elif user == 'rmth':
    logdir = '/scratch/users/rmth/tf_logs/'
    ckptdir = logdir + 'model/'
    data_folder = '/scratch/users/rmth/processed_shhs_data/'

if verbose:
    tf.logging.set_verbosity(tf.logging.INFO)
if not model_memory and os.path.exists(ckptdir):
    rmtree(ckptdir)
if not os.path.exists(logdir):
    os.mkdir(logdir)

(n_channels, n_epoch_samples, fs) = determine_data_dimensions(data_folder)

params= {'logdir': logdir,
         'n_epoch_samples': n_epoch_samples,
         'time_steps': 5*2*6,
         'fs': 128,
         'n_filters': 512,
         'temporal_kernel_size': 3,
         'n_channels': n_channels,
         'pooling_size': 3,
         'rnn_size': 32,
         'n_units_dense': 128,
         'n_classes': 2,
         'batch_norm': True,
         'activation': tf.nn.elu,
         'dropout': True,
         'dropout_pct': .25,
         'kernel_initializer': None,
         'kernel_regularizer': None, #tf.contrib.layers.l2_regularizer(1e-0),
         'regularization': 1e-3,
         'optimizer': tf.train.AdamOptimizer,
         #'optimizer': tf.train.GradientDescentOptimizer,
         #'optimizer': tf.train.AdadeltaOptimizer,
         #'optimizer': tf.train.RMSPropOptimizer,
         'learning_rate': 1e-3,
         'train_pct': .75,
         'val_pct': .5,
         'batch_size': {'train': 8, 'val': 2, 'test': 2},
         'save_checkpoint_steps': 50,
         'save_summary_steps': 50,
         'buffer_size': 10,
         'train_steps': None,
         'eval_steps': 10,
         'verbose_shapes': True,
         'pool_stride': 2,
         'n_layers': 6,
         'rnn_layer': True,
         'dense_layer': True,
         }

if params['n_epoch_samples'] % params['time_steps'] != 0:
    print('Warning: time steps has to divide epoch in equal parts. Quitting.')
    q()

# Setup partitions and batches in the DataHandler-class
DataHandler.setup_partitions(data_folder,
                             params['train_pct'],
                             params['val_pct'],
                             params['batch_size'])
if model_memory:
    with open('partitions.pkl', 'rb') as f:
        DataHandler.set_partitions(pickle.load(f))
else:
    with open('partitions.pkl', 'wb') as f:
        pickle.dump(DataHandler.get_partitions(), f)

config = tf.estimator.RunConfig(save_checkpoints_steps=params['save_checkpoint_steps'],
                                save_summary_steps=params['save_summary_steps'])
model = Model('CRNN', params)

classifier = tf.estimator.Estimator(
        model_fn = lambda features, labels, mode: model(features, labels, mode),
        model_dir = ckptdir,
        config = config)

if train_model:
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn('train', params), max_steps=params['train_steps'])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn('val',params),steps=params['eval_steps'])
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

if evaluate_model:
    # Get predictions for test set
    eval_results = classifier.evaluate(input_fn=lambda: input_fn('test_sequence',params))
    print(eval_results)
    cm = np.asarray([[eval_results["tn"], eval_results["fp"]],
              [eval_results["fn"], eval_results["tp"]]])
    plot_confusion_matrix(cm, classes=['Control', 'Stroke'], normalize=True,
                      title='Normalized confusion matrix')
    plt.show()

if export_savedmodel:
    def serving_input_receiver_fn():
        features = tf.placeholder(shape=[None,params['n_epoch_samples'],params['n_channels']], dtype = tf.float32)
        receiver_tensors = features
        return tf.estimatord.export.TensorServingInputReceiver(features, receiver_tensors)

    classifier.export_savedmodel(export_dir_base=logdir,
                                 serving_input_receiver_fn=serving_input_receiver_fn)

    exports = os.listdir(logdir)
    exports = [int(e) for e in exports if e.isdigit()]
    export_dir = os.path.abspath(logdir + str(exports[np.argmax(exports)]))

    # Test that exported model functions by making prediction and comparing to expectation
    from tensorflow.contrib import predictor
    features, labels = input_fn('train', params)
    with tf.Session() as sess:
        x,y = sess.run([features, labels])
    predict_fn = predictor.from_saved_model(export_dir)
    predictions = predict_fn(
        {"input": x})
    print(np.transpose(predictions['classes']))
    print(np.argmax(y,axis=1))

''' Activation maximization
'''
if gan_activation_maximization:
    import gan_model
    gan = gan_model.GANModel(params)
    gan(input_fn('train', params), gan_train = True)

if activation_maximization:
    print('am')
    #features, labels = input_fn('test_batch',params)
    #predictions = model(features, labels, tf.estimator.ModeKeys.PREDICT).predictions

    def train_prototypes(n_iter = 5000,
                         lr = 1e1,
                         lmdb = 1e-3,
                         lr_decay = 5e-4, # if zero removes decay entirely -> constant lmdb
                         lmdb_decay = 0,
                         tht = 1e-3, #1e-4,  # if zero removes decay entirely -> constant lr
                         yps = 1e-3,
                         ):
        tf.reset_default_graph()
        ckpt = tf.train.get_checkpoint_state(ckptdir)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        sess = tf.Session()

        features, labels = input_fn('train', params)

        saver.restore(sess, ckpt.model_checkpoint_path)
        prototype = tf.get_collection('prototype')

        X_p = prototype[0]
        y_p = prototype[1]
        logits_p = prototype[2]
        cost_p = prototype[3]
        opt_p = prototype[4]
        lr_p = prototype[5]
        lambda_p = prototype[6]
        theta_p = prototype[7]
        X_mean = prototype[8]
        Spectra = prototype[9]
        ypsilon_p = prototype[10]

        f, l = sess.run([features, labels])
        l = np.argmax(l,axis=1)
        mean = np.zeros([params['n_classes'], params['n_channels'], params['time_steps'],1, params['n_epoch_samples']//params['time_steps']])
        spec = np.zeros([params['n_classes'], params['n_epoch_samples'] // params['time_steps'] // 2 +1, params['n_channels']])

        for cls in range(params['n_classes']):
            mean[cls, :,:] = np.reshape ( np.transpose( np.mean( f[cls == l,:,:], axis = 0), [1, 0]), [2,1,1,-1] )
            spec[cls, :,:] = np.mean( np.abs(np.fft.rfft(f[cls == l, :, :], axis = 1))**2, axis= 0)
        spec = np.transpose(spec,[0,2,1])

        for iter in range(n_iter):
            decayed_lr = lr * np.exp(-iter * lr_decay) if lr_decay != 0 else lr
            decayed_lambda_p = lmdb * (np.exp(-iter * lmdb_decay) - np.exp(-iter * 10 * lmdb_decay)) if lmdb_decay != 0 else lmdb
            _, c = sess.run([opt_p, cost_p], feed_dict={lr_p: decayed_lr,
                                                        lambda_p: decayed_lambda_p,
                                                        X_mean: mean,
                                                        theta_p: tht,
                                                        Spectra: spec,
                                                        ypsilon_p: yps})
            if iter % 100 == 0:
                print('Iteration: {:05d} Cost = {:.9f} Learning Rate = {:.9f} Lambda = {:.9f}'.format(iter, c, decayed_lr, decayed_lambda_p))
        prototypes,c = sess.run([X_p,cost_p], feed_dict = {X_mean: mean, Spectra: spec})
        print(c)
        out = sess.run(logits_p)
        print(out)
        print(np.argmax(out, axis=2))

        f,ax = plt.subplots(params['n_classes'],2)
        lim = np.max(prototypes[:])
        for i in range(params['n_classes']):
            for chan in range(params['n_channels']):
                ax[i,chan].plot(prototypes[i,chan,0,0,:])
                if i == 0: ax[i,chan].set_title('Control')
                if i == 1: ax[i,chan].set_title('Stroke')
                ax[i,chan].set_ylabel('EEG' + str(chan+1))
                ax[i,chan].set_xlabel('Samples')
                ax[i,chan].set_ylim([-lim,lim])


''' Extract peaks in Taylor Decomposition
'''

from scipy.ndimage.filters import gaussian_filter as gf
from scipy.signal import find_peaks_cwt
from sklearn.manifold import TSNE
from scipy.signal import welch
if std_peak_analysis:
    tf.reset_default_graph()
    #features, labels = input_fn('test_batch',params)
    features, labels = input_fn('train',params)
    predictions = model(features, labels, tf.estimator.ModeKeys.PREDICT).predictions
    saver = tf.train.Saver()
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(ckptdir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    n_preds = 100
    fs = 128
    std_peaks = []
    std_feats = []
    std_spec = []

    plot_each_peak_extraction = False
    for pred in range(n_preds):
        print(int(pred/n_preds*100))
        preds = None
        x = None
        y = None
        preds, x, y = sess.run([predictions,features,labels])
        x = np.transpose(x, [0, 2, 1])  # channels first
        x = np.reshape(x, [x.shape[0], params['n_channels'], params['time_steps'], 1, -1])
        sens = preds['experimental_sensitivity']
        feat = preds['features']

        #percentile = np.percentile(sens[:], [90])
        percentile = np.max(sens[:])*1e-3
        cols = ['r','b']
        w = 400
        for batch_idx in range(2):
            if plot_each_peak_extraction: f, axarr = plt.subplots(2, 2, sharex=True, squeeze=False)

            for chan in range(2):
                xs = x[batch_idx, chan, 0, 0, :]
                senss = sens[batch_idx, chan, 0, 0, :]

                senss = gf(senss,100)
                idx = find_peaks_cwt(senss,range(10, 20))
                if idx.any(): idx = idx[senss[idx] > percentile]

                for k,j in enumerate(idx):
                    peak = np.zeros([w*2])
                    if j+w > len(xs):
                        continue
                        tmp = xs[j-w:]
                        peak[0:len(tmp)] = tmp
                    elif j-w < 0:
                        continue
                        tmp = xs[:j+w]
                        peak[0:len(tmp)] = tmp
                    else:
                        peak = xs[j-w:j+w]
                    std_peaks.append(peak)
                    std_feats.append(feat)
                    f,Pxx = welch(peak, fs = fs)
                    Pxx = 10*np.log10(Pxx)
                    std_spec.append(Pxx)

                if plot_each_peak_extraction:
                    axarr[0, chan].plot(xs,cols[chan])
                    axarr[1, chan].plot(senss,cols[chan])
                    axarr[1, chan].plot(idx, senss[idx], 'ok')
            if plot_each_peak_extraction: plt.show()

    peaks = np.asarray(std_peaks)
    specs = np.asarray(std_spec)
    feats = [np.ndarray.flatten(e) for e in std_feats]
    feats = np.asarray(feats)



    '''
    from scipy.signal import correlate
    n_peaks = peaks.shape[0]
    window_size = peaks.shape[1]
    X = np.zeros([n_peaks, n_peaks])
    lags = np.zeros([n_peaks, n_peaks]) 
    for i in range(n_peaks):
        for j in range(n_peaks):
            a = peaks[i,:]
            b = peaks[j,:]
            correlation = correlate(a,b,'same')
            #X[i,j] = np.max(correlation)
            X[i, j] = np.sum(np.square(np.abs(correlation)))
            lags[i,j] = np.argmax(correlation) - window_size//2
    tsne = TSNE(n_components=2)
    pc = tsne.fit_transform(X)
    '''

    input = specs #np.concatenate([specs], axis=1)
    #pca = PCA(n_components=50)
    #red = pca.fit_transform(input)

    tsne = TSNE(n_components=2, n_iter = 10000)
    red = tsne.fit_transform(input)

    #pc = pca.fit_transform(feats)

    gm = GaussianMixture(n_components=5, n_init = 10)
    #gm.fit(pc)
    #grp = gm.predict(pc)
    #prop = gm.predict_proba(pc)

    gm.fit(red)
    grp = gm.predict(red)
    prop = gm.predict_proba(red)


    pcxlim = 1.1*np.max(np.abs(red[:,0]))
    pcylim = 1.1*np.max(np.abs(red[:,1]))

    t = np.arange(-w/fs,w/fs, 1/fs)
    f, axarr = plt.subplots(3, 2, sharex=False, squeeze=False)
    axarr[0,0].scatter(red[:,0],red[:,1], c=grp, picker=True)
    axarr[0,0].set_title('tSNE (GMM grouping)')
    axarr[0,1].set_title('GMM centroids')
    for i in range(3):
        for j in range(2):
            axarr[i,j].spines['top'].set_visible(False)
            axarr[i,j].spines['right'].set_visible(False)
    axarr[1,0].set_title('Pick 1')
    axarr[2,0].set_title('Pick 2')

    axarr[0,0].set_xlabel('tSNE 1')
    axarr[0,1].set_xlabel('tSNE 1')
    axarr[0,0].set_ylabel('tSNE 2')
    axarr[0,1].set_ylabel('tSNE 2')
    axarr[0,1].scatter(gm.means_[:,0],gm.means_[:,1],
                       c=range(gm.means_.shape[0]),
                       picker=True)
    for i in range(2):
        axarr[0,i].set_xlim([-pcxlim, pcxlim])
        axarr[0,i].set_ylim([-pcylim, pcylim])
    pick_text = [None,None]

    plt.show()

    def onpick(e):
        axes = e.canvas.figure.axes
        ind = e.ind
        if e.mouseevent.inaxes == axes[0]:
            if len(ind) != 1:
                ind = ind[0]
            ax = 2 if e.mouseevent.button == 1 else 4
            axes[ax].clear()
            axes[ax].plot(t,np.squeeze(peaks[ind,:]))
            axes[ax].set_ylim([-5,5])
            axes[ax].set_ylabel('EEG  [softmax normalized]')
            axes[ax].set_xlabel('Time [s]')
            axes[ax].set_title('Pick ' + str(ax) + ', index: ' + str(ind) + ', group: ' + str(grp[ind]))# + ', prob: ' + str(np.max(prop[ind])) )
            if pick_text[ax//2-1] != None: pick_text[ax//2-1].remove()
            pick_text[ax//2-1] = axes[0].text(red[ind,0],red[ind,1],str(ax//2))

            f,Pxx = welch(np.squeeze(peaks[ind,:]), fs = fs)
            Pxx = 10*np.log10(Pxx)

            axes[ax+1].clear()
            axes[ax+1].plot(f,Pxx)
            axes[ax+1].set_xticks([1, 5, 10, 20])
            axes[ax+1].set_xticklabels(['1','5','10','20'])
            axes[ax+1].set_xlabel('Frequenzy [Hz]')
            axes[ax+1].set_ylabel('Power [dB]')
            axes[ax+1].set_xlim([0,30])
            axes[ax+1].set_ylim([-50,10])
        elif e.mouseevent.inaxes == axes[1]:
            size = np.ones(grp.shape)*8
            size[grp == ind] = 20 if e.mouseevent.button == 1 else 8
            axes[0].collections[0].set_sizes(size)
    cid = f.canvas.mpl_connect('pick_event', onpick)
    f.suptitle('STD Peak Clustering\n\n\n')

''' Visualize Simple Taylor Decomposition/Sensitivity Analysis
'''
if visualize_std:
    features, labels = input_fn('test_batch',params)
    predictions = model(features, labels, tf.estimator.ModeKeys.PREDICT).predictions
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckptdir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        preds, x, y = sess.run([predictions,features,labels])
        x = np.transpose(x, [0, 2, 1])  # channels first
        x = np.reshape(x, [x.shape[0], params['n_channels'], params['time_steps'], 1, -1])
        sens = preds['experimental_sensitivity']
        lim = np.max(np.abs(sens[:]))
        for batch_idx in range(2):
            f, axarr = plt.subplots(params['time_steps'], 2,
                                    sharex=True,
                                    squeeze=False)
            for time_step in range(params['time_steps']):
                axarr[time_step, 0].plot(x[batch_idx, 0, time_step, 0, :],'r')
                axarr[time_step, 0].plot(x[batch_idx, 1, time_step, 0, :],'b')
                axarr[time_step, 1].plot(sens[batch_idx, 0, time_step, 0, :],'r')
                axarr[time_step, 1].plot(sens[batch_idx, 1, time_step, 0, :],'b')
                #axarr[time_step, 1].plot(np.cumsum(sens[batch_idx, 0, time_step, 0, :]),'r')
                #axarr[time_step, 1].plot(np.cumsum(sens[batch_idx, 1, time_step, 0, :]),'b')
                #axarr[time_step, 1].set_ylim([-lim,lim])

                for i in range(2):
                    axarr[time_step, i].spines['top'].set_visible(False)
                    axarr[time_step, i].spines['right'].set_visible(False)
                if time_step == params['time_steps']//2:
                    axarr[time_step, 0].set_ylabel('EEG amplitude')
                    axarr[time_step, 1].set_ylabel('Sensitivity')
                if time_step == 0:
                    axarr[time_step, 0].legend(['Channel 1', 'Channel 2'])
                if time_step == params['time_steps']:
                    axarr[time_step, 1].set_xlabel('Sample')
                    axarr[time_step, 0].set_xlabel('Sample')
            title = 'Control' if y[batch_idx,1] == 0 else 'Stroke'
            f.suptitle(title)
            f.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
            f.subplots_adjust(left=.1, bottom=.1,
                                right=.90, top=.90,
                                wspace=.1, hspace=.1)
        plt.show()


