import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import normalize

rerun_csv_extraction = False
cvd_summary_file = '/home/rasmus/Desktop/shhs-cvd-summary-dataset-0.13.0.csv'
summary_file_1 = '/home/rasmus/Desktop/shhs1-dataset-0.13.0.csv'
if rerun_csv_extraction:
    summary = pd.read_csv(summary_file_1)
    cvd_summary = pd.read_csv(cvd_summary_file)
    summary_idx = ['nsrrid','age_s1', 'bmi_s1', 'Alcoh', 'CgPkYr', 'DiasBP', 'SystBP', 'ahi_a0h4', 'gender']
    cvd_summary_idx = ['nsrrid','afibprevalent', 'prev_stk','stk_date']

    def grouping(row):
        stk_date = row['stk_date']
        if np.isnan(stk_date):
            return 0
        if stk_date < 2 * 365:
            return 1
        else:
            return -1

    cvd_summary_idx.append('stroke')
    cvd_summary['stroke'] = cvd_summary.apply(lambda row: grouping(row), axis=1)

    summary = summary[summary_idx].copy()
    cvd_summary = cvd_summary[cvd_summary_idx].copy()

    d = pd.DataFrame()

    for id in summary['nsrrid']:
            a = summary[summary['nsrrid'] == id]
            b = cvd_summary[cvd_summary['nsrrid'] == id][cvd_summary_idx[1:5]]
            a.reset_index(drop=True, inplace=True)
            b.reset_index(drop=True, inplace=True)
            c = pd.concat([a,b], axis=1)
            d = pd.concat([d,c], axis=0)

    with open('shhs_stroke_comorbidity_data', 'wb') as f:
        pickle.dump(d, f)
else:
    with open('shhs_stroke_comorbidity_data', 'rb') as f:
        df = pickle.load(f)
    red = df.copy()
    red = red.drop('stk_date', 1)
    red = red.drop('stroke', 1)


experimental_ids = df[ df['stroke'] == 1 ]['nsrrid'].copy()
all_control_ids = df[ df['stroke'] == 0 ]['nsrrid'].copy()

n_exp = len(experimental_ids)
n_con = len(all_control_ids)

experimental_ids = np.asarray(experimental_ids)
exps = np.zeros([n_exp, 10])
exps_nsrrid = np.zeros([n_exp, 1])
for counter, id in enumerate(experimental_ids):
    a = red[ df['nsrrid'] == id].as_matrix()
    a[np.isnan(a)] = 0
    exps[counter,:] = a[0,1:]
    exps_nsrrid[counter] = id

cons = np.zeros([n_con, 10])
cons_nsrrid = np.zeros([n_con, 1])
for counter, id in enumerate(all_control_ids):
    a = red[ df['nsrrid'] == id].as_matrix()
    a[np.isnan(a)] = 0
    cons[counter,:] = a[0,1:]
    cons_nsrrid[counter] = id

exps_norm = normalize(exps, axis = 0)
cons_norm = normalize(cons, axis = 0)

def distance(a,b):
    return np.linalg.norm(a-b)

dist = np.zeros([n_exp, n_con])
for e in range(n_exp):
    for c in range(n_con):
        dist[e,c] = distance(exps_norm[e,:],cons_norm[c,:])

choices = np.argmin(dist, axis = 1)
master_dist = dist.copy()
dist_tmp = dist.copy()
iter = 0
printing = False
while True:
    cidx, eidx = np.unique(choices, return_index=True)
    print('***')
    print('{} of {} sorted'.format(len(eidx), len(choices)))
    print('***')
    if len(eidx) == len(choices):
        print('Done in {} iterations.'.format(iter))
        break
    for i in range(n_exp):
        if i in eidx:
            if printing: print('{} good'.format(i))
        else:
            if printing: print('{} bad'.format(i))
            c = choices[i]
            sharers = [lol for lol, x in enumerate(choices == c) if x]
            dist_tmp[:,c] = 10
            alt_choice = np.argmin(dist_tmp, axis=1)
            alt_dist = dist_tmp[sharers, alt_choice[sharers]]
            worst_alt = sharers[np.argmax(alt_dist)]
            choices[sharers] = alt_choice[sharers]
            choices[worst_alt] = c
            if printing:
                print('    Choice is: {}'.format(c))
                print('    Shared with: {}'.format(idx))
                print('    Their distances: {}'.format(dist[idx,c]))
                print('    Alternative distances: {}'.format(alt_dist))
                print('    Worst alternative: {}'.format(worst_alt))
    iter += 1
    if iter == 10:
        print('Did not converge.')
        break

exp_id_matched = exps_nsrrid[eidx]
con_id_matched = cons_nsrrid[cidx]

exp_mus = np.mean(exps, axis = 0)
exp_stds = np.std(exps, axis = 0)

con_mus = np.mean(cons[cidx], axis = 0)
con_stds = np.std(cons[cidx], axis = 0)

labels = red.columns.values

for i in range(exp_mus.shape[0]):
    print('Feature {}:'.format(labels[i+1]))
    print('    exp mu: {}'.format(exp_mus[i]))
    print('    con mu: {}'.format(con_mus[i]))
    print('    exp std: {}'.format(exp_stds[i]))
    print('    con std: {}'.format(con_stds[i]))

controls = all_control_ids.sample(n_exp * 6,
                                  replace=False,
                                  weights=None,
                                  random_state=42)


control_ids = np.asarray(controls)
control_ids = [int(i) for i in control_ids if i not in con_id_matched]

IDs = np.concatenate([control_ids,
                       experimental_ids])
group = np.concatenate([np.zeros(shape=len(control_ids)),
                        np.ones(shape=len(experimental_ids))])

out = pd.DataFrame()
out['IDs'] = IDs
out['group'] = group
out.to_csv('./IDs.csv')

matched_control = pd.DataFrame()
matched_control['conIDs'] = np.reshape(con_id_matched, [-1])
matched_control['expIDs'] = np.reshape(exp_id_matched, [-1])
matched_control.to_csv('./matched_controls.csv')

