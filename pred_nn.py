import warnings
warnings.filterwarnings('ignore')
from datetime import datetime as timedate, timedelta
from scipy.stats import poisson
import joblib
import pickle
from neural_net import *
from sklearn.model_selection import KFold

lags_list = [[7, 1], [28, 1], [56, 1], [7, 7], [7, 28], [28, 7], [28, 28], [56, 7], [56, 28], [7, 56], [28, 56], [56, 56]]
#lags_list = [[28, 1], [49,1], [28, 7], [49, 7], [28,28], [49, 28]] #classification
max_lags = max([l + w + 1 for l, w in lags_list])

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

def make_X(dt):
    X = {"dense": dt[dense_cols].to_numpy()}
    for i, v in enumerate(cat_feats):
        X[v] = dt[[v]].to_numpy()
    return X



useless_cols = ['id', 'date', 'sales', 'd', 'wm_yr_wk', 'weekday', 'label', 'max_sales']

ver=1

submission = pd.read_csv('data/sample_submission.csv')
submission = submission.set_index('id', drop=True)

fday = timedate(2016, 5, 23)

cat_feats = ['dept_id', 'store_id', 'cat_id', 'state_id'] + ['item_id'] + ['year', 'month', 'week', 'wday', 'mday', 'day']

emb_dims = [(7, 1), (10, 2), (3, 1), (3, 1), (3049, 3), (6, 1), (12, 1), (53, 2), (7, 3), (31, 3), (366, 5)]
n_cont = 31

model = M5Net(emb_dims, n_cont).to(device)
#model = mixture(emb_dims, n_cont).to(device)
# dt = joblib.load('data/train_pred_nn.joblib')
# dt['date'] = pd.to_datetime(dt['date'])

for epoch in [10, 12]:

    model_path = 'models/model_%s_%s_%s.pt' % (ver, 1, epoch)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test = joblib.load('data/test_pred_nn.joblib')
    test['date'] = pd.to_datetime(test['date'])
    #test = dt[dt.date >= (fday - timedelta(days=max_lags))]

    train_cols = test.columns[~test.columns.isin(useless_cols)]
    print(train_cols)

    dense_cols = np.setdiff1d(train_cols, cat_feats)

    for day in range(1, 5):

        sub = test.loc[(test['date'] <= fday + timedelta(day * 7 - 1)) & (
                test['date'] >= fday + timedelta((day - 1) * 7)), :]

        print(sub.shape)
        print(sub['date'].unique())

        sub['rmean_56_56'] = sub['rmean_56_56'].fillna(0)
        sub['rmean_56_56'] = sub['rmean_56_56'].fillna(0)

        for var in train_cols:
            print(var, sub[var].isna().sum())

        X_test = make_X(sub[train_cols])
        test_loader = M5Loader(X_test, y=None, cat_cols=cat_feats, batch_size=50000, shuffle=False)

        del X_test
        gc.collect()
        print(gc.collect())

        pred = []

        with torch.no_grad():
            model.eval()
            for i, (X_cont, X_cat, y) in enumerate(tqdm(test_loader)):
                out = model(X_cont, X_cat)
                # out1, out2, out3, out4, out5, out = model(X_cont, X_cat)
                # out_final = (out1+out2+out3+out4+out5+out)/6
                pred += list(out.cpu().numpy().flatten())

        pred = np.array(pred)
        pred = np.where(pred > 1, 1, pred)

        # plt.hist(pred)
        # plt.show()

        test.loc[(test['date'] <= fday + timedelta(day * 7 - 1)) & (
                test['date'] >= fday + timedelta((day - 1) * 7)), 'sales'] = np.asarray(pred)

        if day != 4:
            test = create_lag_features(test)

    test = test[['sales', 'd', 'id']]
    test = pd.pivot_table(test, values='sales', index=['id'], columns=['d'])

    submission.loc[test.index, :] = np.asarray(test.iloc[:, -28:])
    eval_idx = [sub.replace('evaluation', 'validation') for sub in test.index]
    submission.loc[eval_idx, :] = np.asarray(test.iloc[:, -28:])

    sales = pd.read_csv('data/sales_train_evaluation.csv')
    submission.iloc[:30490, -28:] = np.transpose(np.transpose(np.asarray(submission.iloc[:30490, -28:])) * np.transpose(
        np.max(np.asarray(sales.iloc[:, 6:]), axis=1)))

    max_values = np.max(np.asarray(sales.iloc[:, 6:]), axis=1)
    for day in range(28):
        submission.iloc[:30490, day] = np.where(max_values == 0, 0, np.asarray(submission.iloc[:30490, day]))

    submission.to_csv('submission/sub_future_%s_%s.csv' % (epoch, 1))