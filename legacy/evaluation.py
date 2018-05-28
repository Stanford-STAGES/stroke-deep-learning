import pickle
import tensorflow as tf
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn import metrics
from sklearn.utils import resample

with open('probabilities.pkl', 'rb') as f:
    val_group, val_probs, test_group, test_probs, matched_probs, matched_group = pickle.load(f, encoding='latin1')

matched_ids = [e for e in matched_group.keys()]

val_label = []
val_prob = []
for k,v in val_group.items():
    if v:
        val_label.append(v)
        val_prob.append(val_probs[k])
n_exp = len(val_label)

test_label = []
test_prob = []
for k,v in test_group.items():
    if v:
        test_label.append(v)
        test_prob.append(test_probs[k])

matched_label_train = []
matched_prob_train = []
for k, v in matched_group.items():
    if k in matched_ids[:n_exp]:
        matched_label_train.append(v)
        matched_prob_train.append(matched_probs[k])

matched_label_test = []
matched_prob_test = []
for k, v in matched_group.items():
    if k in matched_ids[n_exp:2*n_exp]:
        matched_label_test.append(v)
        matched_prob_test.append(matched_probs[k])


max_length = np.max([len(e) for e in val_prob + matched_prob_train + matched_prob_test])
hidden_size = 1
batch_size_train = 8

x = tf.placeholder(shape=(None, max_length,1), dtype=tf.float32)
y = tf.placeholder(shape=(None, 2), dtype=tf.float32)
prob = tf.placeholder_with_default(1.0, shape=())
batch_size = tf.shape(x)[0]

#lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
#lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)

lstm_fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)
lstm_bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)

lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=prob)
lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=prob)

#lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob = prob)
#lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob = prob)

outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs=x, dtype=tf.float32)
output_rnn = tf.concat(outputs, axis=2)
logit_input = tf.reshape(output_rnn, [batch_size, 2*max_length])
logit = tf.layers.dense(inputs=logit_input,
                         units=2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(cost)
pred = tf.argmax(logit,1)
correct_pred = tf.equal(pred, tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
softmax = tf.nn.softmax(logit)
init = tf.global_variables_initializer()

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p,:]

def gen_data(exp, con, batch_size, max_length):
    n = batch_size // 2
    x = np.zeros(shape = (batch_size, max_length,1))
    y = np.zeros(shape = (batch_size, 2))
    shuffle(exp)
    shuffle(con)
    for i in range(n):
        e = exp[i]
        c = con[i]
        x[i, 0:len(c),0] = c
        y[i,0] = 1
        x[i+n, 0:len(e),0] = e
        y[i+n,1] = 1
    return unison_shuffled_copies(x,y)


x_test = np.zeros(shape=(20, max_length, 1))
y_test = np.zeros(shape=(20, 2))
y_test[0:10,0] = 1
y_test[10:,1] = 1
for i,e in enumerate(matched_prob_test+test_prob):
    n = len(e)
    x_test[i, 0:n, 0] = e

x_val = np.zeros(shape=(20, max_length, 1))
y_val = np.zeros(shape=(20, 2))
y_val[0:10,0] = 1
y_val[10:,1] = 1
for i,e in enumerate(matched_prob_train+val_prob):
    n = len(e)
    x_val[i, 0:n, 0] = e

training_steps = 1000
display_step = 10
with tf.Session() as sess:
    sess.run(init)
    for step in range(1, training_steps + 1):
        batch_x, batch_y = gen_data(val_prob, matched_prob_train, batch_size_train, max_length)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, prob: 0.1})
        if step % display_step == 0 or step == 1:
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
            print('step: {}, loss: {:.2f}, accuracy: {:.2f}'.format(step, loss, acc))
    acc_val, softmax_val = sess.run([accuracy, softmax], feed_dict = {x: x_val, y: y_val, prob: 1.0})
    acc_test, softmax_test = sess.run([accuracy, softmax], feed_dict = {x: x_test, y: y_test, prob: 1.0})

fpr_val, tpr_val, thress_val = roc_curve(np.argmax(y_val,axis=1), softmax_val[:,1], drop_intermediate=False)
fpr_test, tpr_test, thress_test = roc_curve(np.argmax(y_test,axis=1), softmax_test[:,1], drop_intermediate=False)
auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)
f1s_original = []
lol = []
for t in thress_val:
    f1s_original.append(f1_score(y_true=np.argmax(y_val,axis=1), y_pred=np.asarray(softmax_val[:,1] > t, dtype=np.float32)))

n_bootstraps = 100
f1s_bootstraps = np.zeros((len(thress_val), n_bootstraps))
accs_bootstraps = np.zeros((len(thress_val), n_bootstraps))
true_val = np.argmax(y_val,axis=1)
est_val = softmax_val[:,1]
pairs = np.zeros((len(true_val), 2))
pairs[:,0] = true_val
pairs[:,1] = est_val
for j in range(n_bootstraps):
    if j % 10 == 0:
        print(j/n_bootstraps*100)
    pairs_boot = resample(pairs, n_samples = 20, replace=True)
    true_val_boot = pairs_boot[:,0]
    est_val_boot = pairs_boot[:,1]
    for i,t in enumerate(thress_val):
        f1s_bootstraps[i, j] = f1_score(y_true=true_val_boot, y_pred=np.asarray(est_val_boot > t, dtype=np.float32))
        accs_bootstraps[i, j] = accuracy_score(y_true=true_val_boot, y_pred=np.asarray(est_val_boot > t, dtype=np.float32))

f1s_m = np.median(f1s_bootstraps, axis=1)
accs = np.mean(accs_bootstraps, axis=1)

'''
f1s_l = np.percentile(f1s_bootstraps, q=5, axis=1)
f1s_u = np.percentile(f1s_bootstraps, q=95, axis=1)
index = f1s_m/(f1s_u - f1s_l)

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
#best_t_idx = np.nanargmax(index)
best_t_idx = np.nanargmax(f1s_m)
best_t = thress_val[best_t_idx]
test_accuracy = accuracy_score(y_true=np.argmax(y_test,axis=1), y_pred=np.asarray(softmax_test[:,1] > best_t, dtype=np.float32))
tn, fp, fn, tp = metrics.confusion_matrix(np.argmax(y_test, axis=1), softmax_test[:, 1] > best_t).ravel()
test_sensitivity = tp / (tp + fn)
test_specificity = tn / (tn+fp)

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
ax[0].plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([0.0, 1.0])
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].set_title('Test ROC')
ax[0].legend(loc="lower right")

ax[1].plot(thress_val, f1s_m, color='black')

ax[1].plot(thress_val[best_t_idx], f1s_m[best_t_idx], 'or')
ax[1].set_title('F1-scores (validation, bootstrapped)')
ax[1].set_ylim([0.0, 1.0])
ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('f1-score')
plt.suptitle('Test partition ROC')