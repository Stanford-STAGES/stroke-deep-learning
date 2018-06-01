import pickle
import tensorflow as tf
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn import metrics
from sklearn.utils import resample
import os
import traceback
import seaborn as sns
from collections import deque

path = '/home/rasmus/Desktop/c-shhs_cluster_shallowr_cv/'
fold_folders = os.listdir(path)

rocfigs, rocs = plt.subplots(ncols=3, nrows=3)

f1figs, f1s = plt.subplots(ncols=3, nrows=3)
senses_aggr = []
speces_aggr = []
accs_aggr = []

#sm_val = []
#sm_test = []
#ys_val = []
#ys_test = []

for fold in fold_folders:
    try:
        print('Training {}'.format(fold))
        fold_number = int(fold[2])
        with open(path + fold + '/eval/' + 'features.pkl', 'rb') as f:
            train_group, train_feat, val_group, val_feat, test_group, test_feat, matched_feat, matched_group = pickle.load(f, encoding='latin1')

        matched_ids = [e for e in matched_group.keys()]

        train_label_exp = []
        train_feat_exp = []
        train_label_con = []
        train_feat_con = []
        for k, v in train_group.items():
            if v:
                train_label_exp.append(v)
                train_feat_exp.append(train_feat[k])
            else:
                train_label_con.append(v)
                train_feat_con.append(train_feat[k])


        val_label = []
        val_ft = []
        for k,v in val_group.items():
            if v:
                val_label.append(v)
                val_ft.append(val_feat[k])
        n_exp = len(val_label)

        test_label = []
        test_ft = []
        for k,v in test_group.items():
            if v:
                test_label.append(v)
                test_ft.append(test_feat[k])

        matched_label_val = []
        matched_feat_val = []
        shuffle(matched_ids)
        for k, v in matched_group.items():
            if k in matched_ids[:n_exp]:
                matched_label_val.append(v)
                matched_feat_val.append(matched_feat[k])

        matched_label_test = []
        matched_feat_test = []
        for k, v in matched_group.items():
            if k in matched_ids[n_exp:2*n_exp]:
                matched_label_test.append(v)
                matched_feat_test.append(matched_feat[k])

        matched_label_train = []
        matched_feat_train = []
        for k, v in matched_group.items():
            if k in matched_ids[2*n_exp:]:
                matched_label_train.append(v)
                matched_feat_train.append(matched_feat[k])

        _,n_features = train_feat_exp[0].shape
        max_length = np.max([len(e) for e in train_feat_exp + train_feat_con + val_ft + matched_feat_train + matched_feat_test])
        hidden_size = 16
        batch_size_train = 32
        tf.reset_default_graph()
        x = tf.placeholder(shape=(None, max_length, n_features), dtype=tf.float32)
        y = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        prob = tf.placeholder_with_default(1.0, shape=())
        batch_size = tf.shape(x)[0]

        #lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        #lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)


        lstm_fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)
        lstm_bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)


        #lstm_fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(hidden_size)
        #lstm_bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(hidden_size)

        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=prob)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=prob)

        #lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob = prob)
        #lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob = prob)

        outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs=x, dtype=tf.float32)
        output_rnn = tf.concat(outputs, axis=2)
        logit_input = tf.reshape(output_rnn, [batch_size, hidden_size*2*max_length])
        logit = tf.layers.dense(inputs=logit_input,
                                 units=2)

        tv = tf.trainable_variables()
        regularization_cost = 0.2 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y)) + regularization_cost

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
        pred = tf.argmax(logit,1)
        correct_pred = tf.equal(pred, tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        softmax = tf.nn.softmax(logit)
        init = tf.global_variables_initializer()

        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p,:]

        def gen_data(exp, con, batch_size, max_length, n_features):
            n = batch_size // 2
            x = np.zeros(shape = (batch_size, max_length, n_features))
            y = np.zeros(shape = (batch_size, 2))
            shuffle(exp)
            shuffle(con)
            for i in range(n):
                e = exp[i]
                #print('e.shape1: {}'.format(e.shape[0]))
                c = con[i]
                #print('c.shape1: {}'.format(c.shape[0]))
                x[i, 0:c.shape[0], 0:n_features] = c
                y[i,0] = 1

                x[i+n, 0:e.shape[0], 0:n_features] = e
                y[i+n,1] = 1
            return unison_shuffled_copies(x,y)


        x_test = np.zeros(shape=(n_exp*2, max_length, n_features))
        y_test = np.zeros(shape=(n_exp*2, 2))
        y_test[0:n_exp,0] = 1
        y_test[n_exp:,1] = 1
        for i,e in enumerate(matched_feat_test+test_ft):
            if i == 16: break
            n = len(e)
            x_test[i, 0:n, :] = e

        x_val = np.zeros(shape=(n_exp*2, max_length, n_features))
        y_val = np.zeros(shape=(n_exp*2, 2))
        y_val[0:n_exp,0] = 1
        y_val[n_exp:,1] = 1
        for i,e in enumerate(matched_feat_val+val_ft):
            if i == 16: break
            n = len(e)
            x_val[i, 0:n, :] = e

        #training_steps = 300 #1000
        display_step = 20
        buffer_length = 5
        max_step = 3000
        tol = 1e-1
        patience = 2
        with tf.Session() as sess:
            sess.run(init)
            #for step in range(1, training_steps + 1):
            step = 0
            while True:
                step += 1
                #print(step)
                #batch_x, batch_y = gen_data(val_ft, matched_feat_train, batch_size_train, max_length, n_features)
                #batch_x, batch_y = gen_data(train_feat_exp, train_feat_con, batch_size_train, max_length, n_features)
                batch_x, batch_y = gen_data(train_feat_exp, matched_feat_train, batch_size_train, max_length, n_features)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, prob: 0.5})
                if step % display_step == 0 or step == 1:
                    acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
                    acc_val, loss_val = sess.run([accuracy, cost], feed_dict = {x: x_val, y: y_val, prob: 1.0})
                    if step == 1:
                        smoothed_val_losses = deque(np.ones(buffer_length)*loss_val)
                    else:
                        mean1 = np.mean(smoothed_val_losses)
                        smoothed_val_losses.popleft()
                        smoothed_val_losses.append(loss_val)
                        mean2 = np.mean(smoothed_val_losses)
                        diff = mean1-mean2
                        print(diff)
                        print(smoothed_val_losses)
                        print(patience)
                    if step > 2*buffer_length*display_step:
                        if diff < tol:
                            patience -= 1
                        elif patience < 3:
                            patience += 1
                        if patience == 0 or step > max_step:
                            break
                    print('step: {}, train loss: {:.2f}, accuracy: {:.2f}'.format(step, loss, acc))
                    print('step: {}, val loss: {:.2f}, accuracy: {:.2f}'.format(step, loss_val, acc_val))
            #for step in range(10):
            #    batch_x, batch_y = gen_data(train_prob_exp, train_prob_con, batch_size_train, max_length)
            #    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, prob: 0.1})
            acc_val, softmax_val = sess.run([accuracy, softmax], feed_dict = {x: x_val, y: y_val, prob: 1.0})
            acc_test, softmax_test = sess.run([accuracy, softmax], feed_dict = {x: x_test, y: y_test, prob: 1.0})

        #sm_val.append(softmax_val)
        #sm_test.append(softmax_test)
        #ys_val.append(y_val)
        #ys_test.append(y_test)

        fpr_val, tpr_val, thress_val = roc_curve(np.argmax(y_val,axis=1), softmax_val[:,1])
        fpr_test, tpr_test, thress_test = roc_curve(np.argmax(y_test,axis=1), softmax_test[:,1])
        auc_val = auc(fpr_val, tpr_val)
        auc_test = auc(fpr_test, tpr_test)
        f1s_original = []
        for t in thress_val:
            f1s_original.append(f1_score(y_true=np.argmax(y_val,axis=1), y_pred=np.asarray(softmax_val[:,1] > t, dtype=np.float32)))

        n_bootstraps = 300
        thress_val = np.arange(0,1,0.05)
        f1s_bootstraps = np.zeros((len(thress_val), n_bootstraps))
        accs_bootstraps = np.zeros((len(thress_val), n_bootstraps))
        true_val = np.argmax(y_val,axis=1)
        est_val = softmax_val[:,1]
        pairs = np.zeros((len(true_val), 2))
        pairs[:,0] = true_val
        pairs[:,1] = est_val
        for j in range(n_bootstraps):
            if j % 100 == 0:
                print(j/n_bootstraps*100)
            pairs_boot = resample(pairs, n_samples = 20, replace=True)
            true_val_boot = pairs_boot[:,0]
            est_val_boot = pairs_boot[:,1]
            for i, t in enumerate(thress_val):
                tn, fp, fn, tp = metrics.confusion_matrix(true_val_boot, np.asarray(est_val_boot > t, dtype=np.float32)).ravel()
                tpr = tp/(tp+fn)
                fpr = fp/(tn+fp)
                speci = 1-fpr

                f1s_bootstraps[i, j] = 2 * (tpr * speci) / (tpr + speci + np.spacing(0))
                #f1s_bootstraps[i, j] = f1_score(y_true=true_val_boot, y_pred=np.asarray(est_val_boot > t, dtype=np.float32))
                accs_bootstraps[i, j] = accuracy_score(y_true=true_val_boot, y_pred=np.asarray(est_val_boot > t, dtype=np.float32))

        f1s_m = np.median(f1s_bootstraps, axis=1)
        accs = np.mean(accs_bootstraps, axis=1)

        f1s_l = np.percentile(f1s_bootstraps, q=5, axis=1)
        f1s_u = np.percentile(f1s_bootstraps, q=95, axis=1)
        index = f1s_m/(f1s_u - f1s_l + np.spacing(0))

        '''
        fig, ax = plt.subplots(ncols=3)
        ax[0].plot(thress_val, f1s_m)
        ax[0].plot(thress_val, f1s_l,'--')
        ax[0].plot(thress_val, f1s_u,'--')
        ax[1].plot(thress_val, index)
        ax[2].plot(thress_val, f1s_original)
        for i in range(3):
            ax[i].set_xlim([0, 1])
        ax[0].set_ylim([0,1])
        ax[2].set_ylim([0,1])
        '''

        best_t_idx = np.nanargmax(index)
        #best_t_idx = np.nanargmax(f1s_m)
        best_t = thress_val[best_t_idx]
        test_accuracy = accuracy_score(y_true=np.argmax(y_test,axis=1), y_pred=np.asarray(softmax_test[:,1] > best_t, dtype=np.float32))
        tn, fp, fn, tp = metrics.confusion_matrix(np.argmax(y_test, axis=1), softmax_test[:, 1] > best_t).ravel()
        test_sensitivity = tp / (tp + fn)
        test_specificity = 1 - (fp / (tn+fp))


        accs_aggr.append(test_accuracy)
        senses_aggr.append(test_sensitivity)
        speces_aggr.append(test_specificity)
        '''
        fig, ax = plt.subplots(ncols=2)
        lw = 2
        ax[0].plot(fpr_val, tpr_val, '--', color='gray',
                   lw=lw,
                   label='Validation set, AUC: {:.2f}'.format(auc_val))
        ax[0].plot(fpr_test, tpr_test, color='black',
                   lw=lw,
                   label='Test, AUC: {:.2f}, acc.: {:.2f}, sens.: {:.2f}, spec.: {:.2f}'.format(auc_test,
                                                                                                        test_accuracy,
                                                                                                        test_sensitivity,
                                                                                                        test_specificity,))
        '''
        lw = 2
        a = (fold_number - 1) // 3
        b = (fold_number - 1) % 3
        #rocs[a,b].plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        rocs[a,b].set_xlim([0.0, 1.0])
        rocs[a,b].set_ylim([0.0, 1.0])
        rocs[a,b].set_xlabel('False Positive Rate')
        rocs[a,b].set_ylabel('True Positive Rate')
        rocs[a,b].set_title('Test ROC')
        rocs[a,b].legend(loc="lower right")

        #rocs[a,b].plot(fpr_val, tpr_val, '--', color='gray',
        #           lw=lw,
        #           label='Validation set, AUC: {:.2f}'.format(auc_val))
        rocs[a,b].plot(fpr_test, tpr_test, color='black',
                   lw=lw,
                   label='Test, AUC: {:.2f}, acc.: {:.2f}, sens.: {:.2f}, spec.: {:.2f}'.format(auc_test,
                                                                                                test_accuracy,
                                                                                                test_sensitivity,
                                                                                                test_specificity, ))
        rocs[a,b].plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        rocs[a,b].set_xlim([0.0, 1.0])
        rocs[a,b].set_ylim([0.0, 1.0])
        rocs[a,b].set_xlabel('False Positive Rate')
        rocs[a,b].set_ylabel('True Positive Rate')
        rocs[a,b].set_title('Test ROC: cv{}'.format(fold_number))
        rocs[a,b].legend(loc="lower right")
        sns.set()

        f1s[a,b].plot(thress_val, f1s_m, color='black')

        f1s[a, b].plot(thress_val[best_t_idx], f1s_m[best_t_idx], 'or')
        f1s[a, b].set_title('F1-scores (validation, bootstrapped): {}'.format(fold_number))
        f1s[a, b].set_ylim([0.0, 1.0])
        rocs[a,b].set_xlim([0.0, 1.0])
        f1s[a, b].set_xlabel('Threshold')
        f1s[a, b].set_ylabel('f1-score')
        sns.set()

    except Exception as e:
        print('Something went wrong with: {}'.format(fold_number))
        print(e)
        print(traceback.format_exc())


n_bootstraps = 1000
test_accuracy_bootstraps_mean = np.ones(n_bootstraps)
test_accuracy_bootstraps_std = np.ones(n_bootstraps)

test_sens_bootstraps_mean = np.ones(n_bootstraps)
test_sens_bootstraps_std = np.ones(n_bootstraps)

test_spec_bootstraps_mean = np.ones(n_bootstraps)
test_spec_bootstraps_std = np.ones(n_bootstraps)
for j in range(n_bootstraps):
    sample = resample(accs_aggr)
    test_accuracy_bootstraps_mean[j] = np.mean(sample)
    test_accuracy_bootstraps_std[j] = np.std(sample)

    sample = resample(senses_aggr)
    test_sens_bootstraps_mean[j] = np.mean(sample)
    test_sens_bootstraps_std[j] = np.std(sample)

    sample = resample(speces_aggr)
    test_spec_bootstraps_mean[j] = np.mean(sample)
    test_spec_bootstraps_std[j] = np.std(sample)

acc_mean = np.mean(test_accuracy_bootstraps_mean)
acc_std = np.mean(test_accuracy_bootstraps_std)
acc_mean_l = np.percentile(test_accuracy_bootstraps_mean, 2.5)
acc_mean_u = np.percentile(test_accuracy_bootstraps_mean, 97.5)
print('Test accuracy: {:.3} (CI: [{:.3}-{:.3}] +/- {:.3}'.format(acc_mean, acc_mean_l, acc_mean_u, acc_std))
sens_mean = np.mean(test_sens_bootstraps_mean)
sens_std = np.mean(test_sens_bootstraps_std)
print('Test sensitivity: {:.3} +/- {:.3}'.format(sens_mean, sens_std))
spec_mean = np.mean(test_spec_bootstraps_mean)
spec_std = np.mean(test_spec_bootstraps_std)
print('Test specificity: {:.3} +/- {:.3}'.format(spec_mean, spec_std))

plt.show()

'''
for i,l in enumerate(ys_val):
    if i  == 0:
        y_val = l
    else:
        y_val = np.concatenate([y_val,l], axis=0)

for i,l in enumerate(ys_test):
    if i  == 0:
        y_test = l
    else:
        y_test = np.concatenate([y_test,l], axis=0)

for i,l in enumerate(sm_val):
    if i  == 0:
        softmax_val = l
    else:
        softmax_val = np.concatenate([softmax_val,l], axis=0)

for i,l in enumerate(sm_test):
    if i  == 0:
        softmax_test = l
    else:
        softmax_test = np.concatenate([softmax_test,l], axis=0)


fpr_val, tpr_val, thress_val = roc_curve(np.argmax(y_val, axis=1), softmax_val[:, 1], drop_intermediate=False)
fpr_test, tpr_test, thress_test = roc_curve(np.argmax(y_test, axis=1), softmax_test[:, 1], drop_intermediate=False)
auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)
f1s_original = []
lol = []
for t in thress_val:
    f1s_original.append(
        f1_score(y_true=np.argmax(y_val, axis=1), y_pred=np.asarray(softmax_val[:, 1] > t, dtype=np.float32)))

n_bootstraps = 100
f1s_bootstraps = np.zeros((len(thress_val), n_bootstraps))
accs_bootstraps = np.zeros((len(thress_val), n_bootstraps))
true_val = np.argmax(y_val, axis=1)
est_val = softmax_val[:, 1]
pairs = np.zeros((len(true_val), 2))
pairs[:, 0] = true_val
pairs[:, 1] = est_val
for j in range(n_bootstraps):
    if j % 10 == 0:
        print(j / n_bootstraps * 100)
    pairs_boot = resample(pairs, n_samples=20, replace=True)
    true_val_boot = pairs_boot[:, 0]
    est_val_boot = pairs_boot[:, 1]
    for i, t in enumerate(thress_val):
        f1s_bootstraps[i, j] = f1_score(y_true=true_val_boot, y_pred=np.asarray(est_val_boot > t, dtype=np.float32))
        accs_bootstraps[i, j] = accuracy_score(y_true=true_val_boot,
                                               y_pred=np.asarray(est_val_boot > t, dtype=np.float32))

f1s_m = np.median(f1s_bootstraps, axis=1)
accs = np.mean(accs_bootstraps, axis=1)

best_t_idx = np.nanargmax(f1s_m)
best_t = thress_val[best_t_idx]
test_accuracy = accuracy_score(y_true=np.argmax(y_test, axis=1),
                               y_pred=np.asarray(softmax_test[:, 1] > best_t, dtype=np.float32))
tn, fp, fn, tp = metrics.confusion_matrix(np.argmax(y_test, axis=1), softmax_test[:, 1] > best_t).ravel()
test_sensitivity = tp / (tp + fn)
test_specificity = tn / (tn + fp)


fig, ax = plt.subplots(ncols=2)
lw = 2
ax[0].plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([0.0, 1.0])
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].set_title('Test ROC')
ax[0].legend(loc="lower right")

#ax[0].plot(fpr_val, tpr_val, '--', color='gray',
#                lw=lw,
#                label='Validation set, AUC: {:.2f}'.format(auc_val))
ax[0].plot(fpr_test, tpr_test, color='black',
                lw=lw,
                label='Test, AUC: {:.2f}, acc.: {:.2f}, sens.: {:.2f}, spec.: {:.2f}'.format(auc_test,
                                                                                             test_accuracy,
                                                                                             test_sensitivity,
                                                                                             test_specificity, ))
ax[0].plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([0.0, 1.0])
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].set_title('Test ROC, all folds')
ax[0].legend(loc="lower right")

ax[1].plot(thress_val, f1s_m, color='black')

ax[1].plot(thress_val[best_t_idx], f1s_m[best_t_idx], 'or')
ax[1].set_title('F1-scores (validation, bootstrapped)')
ax[1].set_ylim([0.0, 1.0])
ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('f1-score')

plt.show()

fpr_test, tpr_test, thress_test = roc_curve(np.argmax(y_test, axis=1), softmax_test[:, 1], drop_intermediate=False)
auc_test = auc(fpr_test, tpr_test)

n_bootstraps = 100

thresholds = np.arange(0,1,0.01)
fpr_bootstraps = np.zeros((len(thresholds), n_bootstraps))
tpr_bootstraps = np.zeros((len(thresholds), n_bootstraps))
thress_bootstraps = np.zeros((len(thresholds), n_bootstraps))

true_test = np.argmax(y_test, axis=1)
est_test = softmax_test[:, 1]

n_samples = len(true_test)

from sklearn.metrics import confusion_matrix

pairs = np.zeros((n_samples, 2))
pairs[:, 0] = true_test
pairs[:, 1] = est_test
for j in range(n_bootstraps):
    if j % 10 == 0:
        print(j / n_bootstraps * 100)
    pairs_boot = resample(pairs, n_samples=n_samples, replace=True)
    true_test_boot = pairs_boot[:, 0]
    est_test_boot = pairs_boot[:, 1]
    for i,t in enumerate(thresholds):
        tn, fp, fn, tp = confusion_matrix(true_test_boot, est_test_boot > t).ravel()
        fpr_bootstraps[i, j] = fp / (fp + tn)
        tpr_bootstraps[i, j] = tp / (tp + fn)

fpr_test = np.mean(fpr_bootstraps, axis=1)
tpr_test = np.mean(tpr_bootstraps, axis=1)

fig, ax = plt.subplots(ncols=1,squeeze=False)
lw = 2
ax[0,0].plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
ax[0,0].set_xlim([0.0, 1.0])
ax[0,0].set_ylim([0.0, 1.0])
ax[0,0].set_xlabel('False Positive Rate')
ax[0,0].set_ylabel('True Positive Rate')
ax[0,0].set_title('Test ROC')
ax[0,0].legend(loc="lower right")

ax[0,0].plot(fpr_test, tpr_test, color='black',
                lw=lw,
                label='Test, AUC: {:.2f}, acc.: {:.2f}, sens.: {:.2f}, spec.: {:.2f}'.format(auc_test,
                                                                                             test_accuracy,
                                                                                             test_sensitivity,
                                                                                             test_specificity, ))
ax[0,0].plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
ax[0,0].set_xlim([0.0, 1.0])
ax[0,0].set_ylim([0.0, 1.0])
ax[0,0].set_xlabel('False Positive Rate')
ax[0,0].set_ylabel('True Positive Rate')
ax[0,0].set_title('Test ROC, all folds')
ax[0,0].legend(loc="lower right")

plt.show()

'''