import os
import time
import gzip
import openml
import argparse
import numpy as np
from os.path import join, expanduser
import scipy.sparse as sp

import pandas as pd
from openml.datasets import get_dataset

parser = argparse.ArgumentParser(description='openML to vw converter')
parser.add_argument('--skips', type=int, default=0, help='skip some dataset for the subset we are treating?')
parser.add_argument('--do_first', type=int, default=0, help='do first 3rd of total datasets? allows parallelization on multiple computers')
parser.add_argument('--do_mid', type=int, default=0, help='do second 3rd of total datasets? allows parallelization on multiple computers')
parser.add_argument('--do_last', type=int, default=0, help='do last 3rd of total datasets? allows parallelization on multiple computers')
args = parser.parse_args()

# best you export your OPe ML API Key in bash or directly paste it here
OML_API_KEY= os.environ['OML_API_KEY']

if not os.path.exists(cachedir):
    os.makedirs(cachedir)

class Bundle(object):
    def __init__(self, dicko):
        'ad-hoc class to mimic argpase in ipython'
        self.keys = [k for k, v in dicko.items()]
        for var, val in dicko.items():
            object.__setattr__(self, var, val)

openml.config.apikey = os.environ['OML_API_KEY']
openml.config.set_cache_directory(cachedir)

# list and download datasets
"""
See: https://github.com/openml/openml-python/blob/develop/examples/30_extended/datasets_tutorial.py
"""
openml_list = openml.datasets.list_datasets()  # returns a dict

# Show a nice table with some key data properties
datalist = pd.DataFrame.from_dict(openml_list, orient="index")

""
data_ge_3 = datalist.query("NumberOfClasses > 2")
ge3 = data_ge_3[['did']].to_numpy().tolist()
ge3 = [x[0] for x in ge3]

print(f'There are {len(ge3)} datasets with classes > 2. \n\nTheir IDs go like this: \n {ge3}')

# this from alberto's cb_bakeoff python script
def save_vw_dataset(X, y, did, ds_dir):
    n_classes = y.max() + 1
    fname = 'ds_{}_{}.vw.gz'.format(did, n_classes)
    with gzip.open(join(ds_dir, fname), 'w') as f:
        if sp.isspmatrix_csr(X):
            n_classes = y.max() + 1
            for i in range(X.shape[0]):
                f.write('{} | {}\n'.format(y[i] + 1, ' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in zip(X[i].indices, X[i].data))).encode('utf-8')
                       )
        else:
            for i in range(X.shape[0]):
                to_write= '{} | {}\n'.format(y[i] + 1, ' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in enumerate(X[i])
                    if (not np.isnan(val)) and val!=0.0))
                f.write(to_write.encode('utf-8')) # py3 must encode to byte


def shuffle(X, y):
    n = X.shape[0]
    perm = np.random.permutation(n)
    X_shuf = X[perm, :]
    y_shuf = y[perm]
    return X_shuf, y_shuf

if args.do_first:
    # these ones were already saved
    skips=()
    ge3 = ge3[:len(ge3)//3]
    print('doing 1/3 of len(dataset)', ge3)
    print()
elif args.do_mid:
    # these ones were already saved
    ge3 = ge3[len(ge3)//3:2*len(ge3)//3]
    print('doing 2/3  of len(dataset)', ge3)
    print()
elif args.do_last:
    skips=(1559,1560,1565,1567,1568,1569,1596,23380,40474,40475,40476,\
            40477,40478,40496,40497,40498,40499,40516,40519,40520,40663,\
            40664,40668,40670,40671,40672,40677,40678,40682,40685,40686,\
            40687,40691,40700,40707,40708,40709,40711,40923,40926,4153,\
            4340,4538,4541,4552)
    # these ones were already saved
    ge3 = ge3[2*len(ge3)//3:]
    if skips:
        ge3=sorted(list(set(ge3).difference(skips)) + [skips[-1]])
    print('doing 3/3  of len(dataset)', ge3)
    print()
else:
    print('partition of data being done not understood')

new_skips = []
for did in sorted(ge3):
    print(f'processing did: {did}')
    try:
        ds = get_dataset(did, download_data=False)
        X, y, categorical_indicator, attribute_names = ds.get_data(
            target=ds.default_target_attribute, dataset_format="array"
            )
        X, y = shuffle(X, y)
    except Exception as e:
        print(f'{did} data exception: {e}; skipping')
        new_skips.append(did)
        continue
    save_vw_dataset(X, y, did, cachedir)