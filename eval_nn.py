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

useless_cols = ['id', 'date', 'sales', 'd', 'wm_yr_wk', 'weekday', 'label', 'max_sales']

ver=1

submission = pd.read_csv('data/sample_submission.csv')
submission = submission.set_index('id', drop=True)

bs = 50000
shuffle = True
epochs = 20
lr = 0.001
fday = timedate(2016, 5, 23)


test = joblib.load('data/test_nn.joblib')
test['date'] = pd.to_datetime(test['date'])
test = test.loc[test['date']>='2016-05-23']

dt = joblib.load('data/train_nn.joblib')

dt = pd.concat([dt, test], axis=0)

del test
gc.collect()
print(gc.collect())

train_cols = dt.columns[~dt.columns.isin(useless_cols)]
print(train_cols)

cat_feats = ['dept_id', 'store_id', 'cat_id', 'state_id'] + ['item_id'] + ['year', 'month', 'week', 'wday', 'mday', 'day']
TARGET = 'sales'

uniques = []
for i, v in enumerate(tqdm(cat_feats)):
    dt[v] = OrdinalEncoder(dtype="int").fit_transform(dt[[v]])
    uniques.append(len(dt[v].unique()))

dense_cols = np.setdiff1d(train_cols, cat_feats)

def make_X(dt):
    X = {"dense": dt[dense_cols].to_numpy()}
    for i, v in enumerate(cat_feats):
        X[v] = dt[[v]].to_numpy()
    return X

dt['date'] = pd.to_datetime(dt['date'])
test = dt.loc[dt['date']>=(fday - timedelta(days=max_lags))]

joblib.dump(test, 'data/test_pred_nn.joblib')

del test
gc.collect()

dt = dt.loc[dt['date']<'2016-05-23']
joblib.dump(dt, 'data/train_pred_nn.joblib')

model_path = 'models/model_%s_%s_%s.pt' % (ver, 1, epochs)

if os.path.isfile(model_path):
    print('model is here')

    del dt
    gc.collect()
    print(gc.collect())

else:

    #joblib.dump(test, 'data/test_pred_nn.joblib')

    # del test
    # gc.collect()
    # print(gc.collect())

    train_index, val_index = train_test_split(dt.index, test_size=0.05, random_state=42, shuffle=True)
    #train_index, val_index = dt.loc[dt['date']<'2016-02-28'].index, dt.loc[dt['date'] >= '2016-02-28'].index

    val = dt.loc[val_index]
    train = dt

    del dt, train_index, val_index
    gc.collect()
    print(gc.collect())

    X_train = make_X(train[train_cols])
    y_train = train[TARGET]
    #y_lbl = train['label']

    del train
    gc.collect()
    print(gc.collect())

    validx, validy = make_X(val[train_cols]), val[TARGET]

    del val
    gc.collect()
    print(gc.collect())

    train_loader = M5Loader(X_train, y_train.values, cat_cols=cat_feats, batch_size=bs, shuffle=shuffle)

    del X_train
    gc.collect()
    print(gc.collect())

    val_loader = M5Loader(validx, validy.values,  cat_cols=cat_feats, batch_size=bs, shuffle=shuffle)

    del validx
    gc.collect()
    print(gc.collect())

    # cat_feats = ['idx', 'dept_id', 'store_id', 'cat_id', 'state_id'] + ['item_id'] + ['year', 'month', 'week', 'wday', 'mday', 'day']
    dims = [1, 2, 1, 1, 3, 1, 1, 2, 3, 3, 5]
    emb_dims = [(x, y) for x, y in zip(uniques, dims)]
    print(emb_dims)
    n_cont = train_loader.n_conts
    print(n_cont)


    model = M5Net(emb_dims, n_cont).to(device)
    print(model)
    criterion = ZeroBalance_RMSE()
    #cl_criterion = torch.nn.BCELoss()

    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, [8, 12, 15], gamma=0.5,
                        last_epoch=-1)

    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=5, verbose=True)
    for epoch in tqdm(range(epochs)):

        model_path = 'models/model_%s_%s_%s.pt' % (ver, 1, epoch)

        # if epoch>=5:
        #     model_path = 'models/model_%s_%s_%s.pt' % (epoch, ver, 1)

        train_loss, val_loss = 0, 0

        # Training phase
        model.train()
        bar = tqdm(train_loader)

        for i, (X_cont, X_cat, y) in enumerate(bar):

            optimizer.zero_grad()
            out = model(X_cont, X_cat)
            loss = criterion(out, y) #+ 0.5 * cl_criterion(out_lbl, y_lbl.to(device))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += loss.item() / len(train_loader)
                bar.set_description(f"{loss.item():.3f}")

        # print(f"Running Train loss: {train_loss}")

        # Validation phase
        with torch.no_grad():
            model.eval()
            for phase in ["valid"]:
                rloss = 0
                if phase == "train":
                    loader = train_loader
                else:
                    loader = val_loader

                y_true = []
                y_pred = []

                for i, (X_cont, X_cat, y) in enumerate(loader):
                    out = model(X_cont, X_cat)
                    loss = criterion(out, y) #+ 0.5 * cl_criterion(out_lbl, y_lbl.to(device))
                    rloss += loss.item() / len(loader)
                    y_pred += list(out.detach().cpu().numpy().flatten())
                    y_true += list(y.cpu().numpy())

                rrmse = rmse_metric(y_pred, y_true)
                print(f"[{phase}] Epoch: {epoch} | Loss: {rloss:.4f} | RMSE: {rrmse:.4f}")

        early_stopping(rrmse, model, path=model_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        train_losses.append(train_loss)
        val_losses.append(rloss)
        scheduler.step()

    del train_loader, val_loader
    gc.collect()
    print(gc.collect())

    torch.save(model, model_path)

'''
model = torch.load(model_path)
model.eval()

# test = joblib.load('data/test_pred_nn.joblib')
# test['date'] = pd.to_datetime(test['date'])

#model.load_state_dict(torch.load(model_path))
for day in range(1, 5):

    sub = test.loc[(test['date'] <= fday + timedelta(day * 7 - 1)) & (
                test['date'] >= fday + timedelta((day - 1) * 7)), :]

    for var in train_cols:
        print(var, sub[var].isna().sum())

    X_test = make_X(sub[train_cols])
    test_loader = M5Loader(X_test, y=None, cat_cols=cat_feats, batch_size=bs, shuffle=False)

    del X_test
    gc.collect()
    print(gc.collect())

    pred = []
    with torch.no_grad():
            model.eval()
            for i, (X_cont, X_cat, y) in enumerate(tqdm(test_loader)):
                out = model(X_cont, X_cat)
                pred += list(out.cpu().numpy().flatten())
    pred = np.array(pred)
    pred = np.where(pred>1, 1, pred)

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
submission.iloc[:30490, -28:] = np.transpose(np.transpose(np.asarray(submission.iloc[:30490, -28:]))*np.transpose(np.max(np.asarray(sales.iloc[:, 6:]), axis=1)))

max_values = np.max(np.asarray(sales.iloc[:, 6:]), axis=1)
for day in range(28):
    submission.iloc[:30490, day] = np.where(max_values==0, 0, np.asarray(submission.iloc[:30490, day] ))

submission.to_csv('submission/sub_nn_%s_%s.csv' %(ver, 1))
'''