import pandas as pd
import numpy as np
summary_file = '/home/rasmus/Desktop/shhs-cvd-summary-dataset-0.13.0.csv'

df = pd.read_csv(summary_file)
def grouping(row):
    stk_date = row['stk_date']
    if np.isnan(stk_date):
        return 0
    if stk_date < 2*365:
        return 1
    else:
        return -1
df['stroke'] = df.apply( lambda row: grouping(row), axis = 1)

ids = df['nsrrid']
stk_date = df['stk_date']
stk_date[ np.isnan(stk_date) ] = 1e6
experimental_ids = ids[stk_date < 2*365]
all_control_ids = ids[stk_date == 1e6]
n_exp = len(experimental_ids)
controls = df[ df['stroke'] == 0]
controls = df.sample(n_exp*5,
                        replace = False,
                        weights = None,
                        random_state = 42)

control_ids = np.asarray(controls['nsrrid'])
experimental_ids = np.asarray(experimental_ids)

IDs = np.concatenate([control_ids,
                       experimental_ids])
group = np.concatenate([np.zeros(shape=len(control_ids)),
                        np.ones(shape=len(experimental_ids))])

out = pd.DataFrame()
out['IDs'] = IDs
out['group'] = group

out.to_csv('./IDs.csv')