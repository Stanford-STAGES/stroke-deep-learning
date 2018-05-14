import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score

with open('probabilities.pkl', 'rb') as f:
    test_group, test_probs, val_group, val_probs = pickle.load(f, encoding='latin1')


'''
fig, ax = plt.subplots(nrows=2)

for k,v in test_group.items():
    if v:
        ax[1].plot(test_probs[k])
    else:
        ax[0].plot(test_probs[k])
for k,v in val_group.items():
    if v:
        ax[1].plot(val_group[k])
    else:
        ax[0].plot(val_group[k])
'''


fig, ax = plt.subplots(ncols=2)
#index = lambda x, thres: np.mean(x)
#index = lambda x, thres: np.mean(x) / np.std(x)
#index = lambda x, thres: np.mean(x > thres)
index = lambda x, thres: sum(x > thres) / len(x)
for thres in np.arange(0.35, .65, 0.025):
    true = []
    est = []
    for k,v in val_group.items():
        true.append(v)
        est.append(index(val_probs[k], thres))


    lw = 1
    fpr, tpr, thress = roc_curve(true, est)
    roc_auc = auc(fpr, tpr)
    f1s = []
    accs = []
    for t in thress:
        f1s.append(f1_score(y_true = true, y_pred = np.asarray(est > t, dtype=np.float32)))
        accs.append(accuracy_score(y_true = true, y_pred = np.asarray(est > t, dtype=np.float32)))
    best_t_idx = np.argmax(f1s)
    best_t = thress[best_t_idx]

    true = []
    est = []
    for k, v in test_group.items():
        true.append(v)
        est.append(index(test_probs[k], thres))
    fpr_test, tpr_test, thress_test = roc_curve(true, est)
    roc_auc_test = auc(fpr_test, fpr_test)
    test_acc = accuracy_score(y_true=true, y_pred=np.asarray(est > best_t, dtype=np.float32))

    ax[0].plot(fpr, tpr, color='black',
             lw=lw,
            label='AUC: {:.2f}, test: acc.: {:.2f}, sens.: {:.2f}, spec.: {:.2f}'.format(roc_auc,test_acc, tpr_test[9], 1-fpr_test[9]))
    #ax[0].plot(fpr_test, tpr_test, '--', color='black',
    #         lw=lw, label='Test:AUC: {:.2f}, acc.: {:.2f}'.format(roc_auc_test,test_acc))
    ax[0].plot(fpr[best_t_idx], tpr[best_t_idx], 'or')
    ax[0].text(fpr[best_t_idx], tpr[best_t_idx], '{:.2f}'.format(test_acc))
    ax[0].plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC')
    ax[0].legend(loc="lower right")
    ax[1].plot(thress, f1s, color='black', label='thress=%0.2f)' % thres)
    ax[1].plot(thress[best_t_idx], f1s[best_t_idx], 'or')
    ax[1].set_title('F1-score')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_xlabel('Threshold')
    ax[1].set_ylabel('f1-score')
    plt.suptitle('Validation partition ROC')


plt.show()
