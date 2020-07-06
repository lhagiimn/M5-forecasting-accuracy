import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib
import pickle
import lightgbm as lgb
import gc
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

from sklearn.model_selection import KFold

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

params_k = {
            'boosting_type': 'gbdt',
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'metric': 'rmse',
            'subsample': 0.5,
            'subsample_freq': 1,
            'learning_rate': 0.03,
            'num_leaves': 2**11-1,
            'min_data_in_leaf': 2**12-1,
            'feature_fraction': 0.5,
            'max_bin': 100,
            'n_estimators': 1500,
            'boost_from_average': False,
            "random_seed":42,
                }




################ lightGBM

useless_cols = ['id', 'date', 'sales', 'd', 'wm_yr_wk', 'weekday']

ver = 'lag_49'
epoch=1500

dt = joblib.load('data/train_lgb.joblib')


# df_lbl = pd.concat([joblib.load('data/train_lbl_1.joblib'),
#                     joblib.load('data/train_lbl_2.joblib'),
#                     joblib.load('data/train_lbl_3.joblib'), ], axis=0, sort=False)
# joblib.dump(df_lbl, 'data/df_lbl.joblib')

# df_lbl = joblib.load('data/df_lbl.joblib')
# dt = dt.merge(df_lbl, on = ['id', 'd'], copy=False)
#
# del df_lbl
# gc.collect()
# print(gc.collect())



train_cols = dt.columns[~dt.columns.isin(useless_cols)]
print(train_cols)

cat_feats = ['dept_id', 'store_id', 'cat_id', 'state_id'] + ['item_id'] #+ ['dig']
TARGET = 'sales'

cat_feats = [f for f in cat_feats if f in train_cols]

train_index, val_index = train_test_split(dt.index, test_size=0.05, random_state=42, shuffle=True)
train_set, val_set = dt.loc[train_index], dt.loc[val_index]

del dt, train_index, val_index
gc.collect()
print(gc.collect())

print(cat_feats)

train_data = lgb.Dataset(data=train_set[train_cols],
                         label=train_set[TARGET],
                         categorical_feature=cat_feats,
                         free_raw_data=False)
print(train_data)
del train_set
gc.collect()
print(gc.collect())

valid_data = lgb.Dataset(data=val_set[train_cols],
                         label=val_set[TARGET],
                         categorical_feature=cat_feats,
                         free_raw_data=False)

del val_set
gc.collect()
print(gc.collect())


model_gbm = lgb.train(params_k, train_data, valid_sets=[valid_data],
                  num_boost_round=epoch, early_stopping_rounds=25,
                  verbose_eval=25)

with open('models/model_lgb_%s_%s.pkl' %(ver, 1), 'wb') as fout:
    pickle.dump(model_gbm, fout)

