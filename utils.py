import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec

import seaborn as sns; sns.set()

from datetime import datetime, timedelta
from tqdm import tqdm
import gc

from IPython.core.display import display, HTML

SEED = 42             # Our random seed for everything
# random.seed(SEED)     # to make all tests "deterministic"
np.random.seed(SEED)


def load_cal():
    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
             "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int8",
            "month": "int8", "year": "int16", "snap_CA": "int8", 'snap_TX': 'int8', 'snap_WI': 'int8' }

    cal = pd.read_csv('data/calendar.csv', parse_dates=['date'], dtype = CAL_DTYPES)

    for col in ['event_name_1', 'event_type_1','event_name_2', 'event_type_2']:   # filling NA
        cal[col].cat.add_categories(['no_event'], inplace=True)
        cal[col].fillna('no_event', inplace=True)

    # wday scheme is: Sat=1..Fri=7
    # cal['wday'] = (cal.wday + 5) % 7 # convert to Mon=1..Sun=7 scheme
    # del cal['weekday']  # consider dropping calendar.weekday

    print("Calendar shape:", cal.shape)
    return cal

def load_sales(cal):
    SALES_DTYPES = {d:"int32" for d in cal.d.unique()}
    SALES_DTYPES = {'item_id':'category', 'dept_id':'category', 'cat_id':'category',
                    'store_id':'category', 'state_id':'category', **SALES_DTYPES}

    sales = pd.read_csv('data/sales_train_evaluation.csv', dtype = SALES_DTYPES)  # replace with sales_train_evaluation.csv later
    print("Sales shape:", sales.shape)
    # consider deleting '_validation' suffix
    return sales

def load_prices():
    PRICES_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
    prices = pd.read_csv('data/sell_prices.csv', dtype = PRICES_DTYPES)
    print("Prices shape:", prices.shape)
    return prices

def load_sub():
    submission = pd.read_csv('data/sample_submission.csv')
    submission.iloc[:,1:] = submission.iloc[:,1:].astype('int8')
    print("Submission shape:", submission.shape)
    return submission





