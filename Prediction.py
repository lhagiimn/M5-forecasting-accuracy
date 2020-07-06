from utils import *
import joblib
import pickle
import matplotlib.pylab as plt

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error


lags_list = [[7, 1], [28, 1], [7, 7], [7, 28], [28, 7], [28, 28]]
max_lags = max([l + w + 1 for l, w in lags_list])
print(max_lags)
# lags encoded as [lag, interval]

def create_lag_features(dt):
    tmp_columns = []
    for lag, window in tqdm(lags_list, leave=False, desc='Lag features'):

        if window == 1:
            lag_col = f"lag_{lag}"
            print(lag_col, end=', ')
            dt[lag_col] = dt[["id", "sales"]].groupby("id")["sales"].shift(lag).astype('float32')

        elif window > 1:
            lag_col = f"lag_{lag}"
            if lag_col not in dt.columns:
                dt[lag_col] = dt[["id", "sales"]].groupby("id")["sales"].shift(lag).astype('float32')
                tmp_columns.append(lag_col)

            rmean_col = f"rmean_{lag}_{window}"
            print(rmean_col, end=', ')
            dt[rmean_col] = dt[["id", lag_col]].groupby("id")[lag_col]. \
                transform(lambda x: x.rolling(window).mean()).astype('float32')

    print('dropping tmp cols:', tmp_columns)
    dt.drop(tmp_columns, axis=1, inplace=True)
    return dt

fday = datetime(2016, 5, 23)
min_lag = 7


submission = pd.read_csv('data/sample_submission.csv')
submission = submission.set_index(['id'], drop=True)
# eval = pd.read_csv('data/sales_train_evaluation.csv')
# submission.iloc[:30490, :] = eval.iloc[:, -28:]
# submission.iloc[30490:, :] = eval.iloc[:, -28:]
#submission.to_csv('sub_call.csv')


useless_cols = ['id', 'date', 'sales', 'd', 'wm_yr_wk', 'weekday']


dt = joblib.load('data/test_lgb.joblib')
ver = 'lag_49'

dt['date'] = pd.to_datetime(dt['date'])

# test_lbl = joblib.load('data/test_lbl_3.joblib')
# dt = dt.merge(test_lbl, on = ['id', 'd'], copy=False)

for fold in [1]:

    with open('models/model_lgb_1_1.pkl', 'rb') as fin:
        model_gbm = pickle.load(fin)

    # fig, ax = plt.subplots(figsize=(12, 8))
    # model_gbm.plot_importance(model_gbm, max_num_features=100, height=0.8, ax=ax)
    # ax.grid(False)
    # plt.title("LightGBM - Feature Importance", fontsize=15)
    # plt.show()

    #dt = create_lag_features(dt)
    #train_cols = dt.columns[~dt.columns.isin(useless_cols)]
    train_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday', 'month',
       'year', 'E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9',
       'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19',
       'E20', 'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29',
       'E30', 'ET0', 'ET1', 'ET2', 'ET3', 'ET4', 'sell_price', 'snap', 'lag_7',
       'lag_28', 'rmean_7_7', 'rmean_7_28', 'rmean_28_7', 'rmean_28_28']

    cat_feats = ['dept_id', 'store_id', 'cat_id', 'state_id'] + ['item_id'] + ['dig']

    cat_feats = [f for f in cat_feats if f in train_cols]
    print(f"train features: {len(train_cols)}, cat features: {len(cat_feats)}")

    for epoch in [1450]:
        df= dt
        for day in range(1, 5):
            sub = df.loc[(df['date']<=fday + timedelta(day*min_lag-1)) & (df['date']>=fday + timedelta((day-1)*min_lag)), :]

            for var in train_cols:
                print(var, sub[var].isna().sum())

            pred = model_gbm.predict(sub[train_cols])
            pred = np.where(pred<0, 0, pred)

            # diff = np.asarray(sub['max_sales'])-pred
            # pred = np.where(diff<0,  np.asarray(sub['max_sales']), pred)

            # plt.plot(pred)
            # plt.show()

            df.loc[(df['date']<=fday + timedelta(day*min_lag-1)) & (df['date']>=fday + timedelta((day-1)*min_lag)), 'sales'] = np.asarray(pred)

            if day!=4:
                df = create_lag_features(df)

            # plt.plot(df.loc[(df['date']<=fday + timedelta(day*7-1)) & (df['date']>=fday + timedelta((day-1)*7)), 'sales'])
            # plt.show()

        test = df[['sales', 'd', 'id']]

        test = pd.pivot_table(test, values='sales', index=['id'], columns=['d'])

        eval_idx = [sub.replace('validation', 'evaluation') for sub in test.index]


        submission.loc[test.index, :] = np.asarray(test)[:, -28:]
        submission.loc[eval_idx, :] = np.asarray(test)[:, -28:]

        submission.to_csv('submission/sub_future_%s_%s.csv' %(epoch, fold))


