from utils import *
import joblib
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import poisson


cal = load_cal()
sales = load_sales(cal)
val_inds = np.random.choice(sales.index.values, 1000, replace=False)
sales = sales.loc[val_inds, :]
sales.iloc[:, 6:] = np.transpose(np.transpose(np.asarray(sales.iloc[:, 6:]))/np.transpose(np.max(np.asarray(sales.iloc[:, 6:]), axis=1)))

prices = load_prices()
prices['sell_price'] = np.asarray(prices['sell_price'])/np.max(np.asarray(prices['sell_price']))

submission = load_sub()

start_day = 1  # set to 1 to capture whole history
tr_last = 1941  # last avaialble sales date
test_last = 1969  # last day of test period   # UPDATE TO 1969 IN STAGE 2
fday = datetime(2016, 5, 23)  # first day of forecast period

catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
numcols = [f"d_{day}" for day in range(start_day,test_last+1)]

sales_tmp = sales[:]
sales_tmp = sales_tmp.drop(sales_tmp.columns[6:6+start_day-1], axis=1)

for day in range(tr_last+1, test_last+1):  # add NANs for forecast sales
    if f"d_{day}" not in sales_tmp.columns:
        sales_tmp[f"d_{day}"] = np.nan


fix_xmas = False  # move down to use calendar feature E1

if fix_xmas:
    print ('Xmas fix')
    xmas_days = cal[cal.E1==1].d.unique()

    for d in xmas_days:
        d1 = int(d[2:])-1
        d1 = 'd_'+ str(d1)
        print (d, d1)
        sales_tmp.loc[:,d] = sales_tmp.loc[:,d1]

df = pd.melt(sales_tmp, id_vars=catcols,
             value_vars=numcols, var_name="d", value_name="sales")

df.sales=df.sales.astype('float32')
del sales_tmp #(100 MB)

# calendar fixes
events = cal.event_name_1.cat.categories.values.tolist()
e_types = cal.event_type_1.cat.categories.values.tolist()
print(events)
print(e_types)

for e in range(len(events)):
    cal[f'E{e}'] = 0
    cal[f'E{e}'] = cal[f'E{e}'].astype('int8')
    cal.loc[(cal.event_name_1 == events[e]) | (cal.event_name_2 == events[e]), f'E{e}'] = 1

for et in range(len(e_types)):
    cal[f'ET{et}'] = 0
    cal[f'ET{et}'] = cal[f'ET{et}'].astype('int8')
    cal.loc[(cal.event_type_1 == e_types[et]) | (cal.event_type_2 == e_types[et]), f'ET{et}'] = 1

cal = cal.drop(['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'], axis=1)

cal_nba_fix = False

if cal_nba_fix:
    cal['nba'] = (cal.E16+cal.E17).cumsum()%2
    fig, ax = plt.subplots(figsize=(18,1))
    del cal['E16'], cal['E17']
    cal.nba.plot()
    plt.show()

cal_lent_fix = False

if cal_lent_fix:
    cal['lent']= (cal.E11+cal.E4).cumsum()%2
    fig, ax = plt.subplots(figsize=(18,1))
    del cal['E11'], cal['E12']
    cal.lent.plot()
    plt.show()

df = df.merge(cal, on="d", copy=False)

# add 'dig' categorical feature with last digit of price tag.
price_dig_add = False  # negative 0.0045 LB effect

if price_dig_add:
    prices['dig']=(prices.sell_price*100+0.01).astype('int').astype('str').str[-1].astype('category')

df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False, how='left')  # add prices
# match SNAP with states
df_snap_fix = True

if df_snap_fix:
    for state in df.state_id.unique():
        df.loc[(df.state_id==state)&(df['snap_'+state]==1),'snap'] = 1
        print (state, end=', ')
    df.snap = df.snap.fillna(0).astype('int8')
    df = df.drop(['snap_'+state for state in df.state_id.unique()], axis=1)


#lags_list = [[7, 1], [28, 1], [7, 28], [28, 7], [7, 28], [28, 28]]
lags_list = [[7, 1], [28, 1], [56, 1], [7, 7], [7, 28], [28, 7], [28, 28], [56, 7], [56, 28], [7, 56], [28, 56], [56, 56]]
#lags_list = [[28, 1], [56,1], [28, 7], [56, 7], [28,28], [56, 28], [28,56], [56, 56]] #classification
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

def create_date_features(dt):
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        #"quarter": "quarter",
        "year": "year",
        "mday": "day",
        'day': "dayofyear"
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(
                dt["date"].dt, date_feat_func).astype("int16")

    dt['freq'] = dt[['id', 'sell_price']].groupby('id')['sell_price'].transform('count').astype(np.int)
    # sum_sales= dt.groupby(['item_id', 'year', 'week'], as_index=False)['sales'].sum()
    # sum_sales = sum_sales.rename(columns={'sales': 'sum_sales'})
    # dt = dt.merge(sum_sales, on=['item_id', 'year', 'week'], copy=False)

    #del sum_sales

    #icols = [
        # ['state_id'],
        # ['store_id'],
        # ['cat_id'],
        # ['dept_id'],
        # ['item_id'],
        # ['state_id', 'cat_id'],
        # ['state_id', 'dept_id'],
        # ['store_id', 'cat_id'],
        # ['store_id', 'dept_id'],
        # ['item_id', 'state_id'],
        #['item_id', 'store_id']
    #]

    #
    # for col in icols:
    #     col_name = '_' + '_'.join(col) + '_'
    #     temp = dt.groupby(col, as_index=False)['sales'].mean()
    #     temp = temp.rename(columns={'sales': 'enc%smean' %col_name})
    #     dt = dt.merge(temp, on=col, copy=False, how='left')
    #     dt['enc%smean' %col_name] = dt['enc%smean' %col_name].fillna(0)

    count_zero = dt.loc[dt['sales']==0, ['id', 'sell_price']].groupby('id', as_index=False)['sell_price'].count()
    count_zero = count_zero.rename(columns={'sell_price': 'sparse_0'})
    dt = dt.merge(count_zero, on=['id'], copy=False, how='left')
    dt['sparse_0'] = dt['sparse_0'].fillna(0)
    dt['sparse_0'] = dt['sparse_0'] / dt['freq']

    del count_zero
    #
    # count_zero = dt.loc[(dt['sales'] >0) & (dt['sales'] <= 10), ['id', 'sell_price']].groupby('id', as_index=False)['sell_price'].count()
    # count_zero = count_zero.rename(columns={'sell_price': 'sparse_%s' % 10})
    # dt = dt.merge(count_zero, on=['id'], copy=False, how='left')
    # dt['sparse_%s' % 10] = dt['sparse_%s' % 10].fillna(0)
    # dt['sparse_%s' % 10] = dt['sparse_%s' % 10] / dt['freq']
    #
    # del count_zero
    #
    # count_zero = dt.loc[dt['sales'] > 10, ['id', 'sell_price']].groupby('id', as_index=False)['sell_price'].count()
    # count_zero = count_zero.rename(columns={'sell_price': 'sparse_%s' % 'more'})
    # dt = dt.merge(count_zero, on=['id'], copy=False, how='left')
    # dt['sparse_%s' % 'more'] = dt['sparse_%s' % 'more'].fillna(0)
    # dt['sparse_%s' % 'more'] = dt['sparse_%s' % 'more'] / dt['freq']
    #
    # del count_zero

    # item_max = dt[['id', 'sales']].groupby('id', as_index=False)['sales'].max()
    # item_max = item_max.rename(columns={'sales': 'max_sales'})
    # dt = dt.merge(item_max, on=['id'], copy=False)
    #
    # del item_max
    # #
    item_mean = dt[['id', 'sales']].groupby('id', as_index=False)['sales'].mean()
    item_mean = item_mean.rename(columns={'sales': 'mean_sales'})
    dt = dt.merge(item_mean, on=['id'], copy=False)

    del item_mean

    # item_q25 = dt[['id', 'sales']].groupby('id', as_index=False)['sales'].quantile(0.25)
    # item_q25 = item_q25.rename(columns={'sales': 'sales_q25'})
    # dt = dt.merge(item_q25, on=['id'], copy=False)
    #
    # del item_q25
    #
    # item_q75 = dt[['id', 'sales']].groupby('id', as_index=False)['sales'].quantile(0.75)
    # item_q75 = item_q75.rename(columns={'sales': 'sales_q75'})
    # dt = dt.merge(item_q75, on=['id'], copy=False)
    #
    # del item_q75

    # item_q90 = dt[['id', 'sales']].groupby('id', as_index=False)['sales'].quantile(0.90)
    # item_q90 = item_q90.rename(columns={'sales': 'sales_q90'})
    # dt = dt.merge(item_q90, on=['id'], copy=False)
    #
    # del item_q90

    # weekly_mean = dt[['month', 'week', 'sales']].groupby(['month', 'week'], as_index=False)['sales'].mean()
    # weekly_mean = weekly_mean.rename(columns={'sales': 'weekly_mean'})
    # dt = dt.merge(weekly_mean, on=['month', 'week'], copy=False)
    #
    # del weekly_mean

    # daily_mean = dt[['month', 'mday', 'sales']].groupby(['month', 'mday'], as_index=False)['sales'].mean()
    # daily_mean = daily_mean.rename(columns={'sales': 'daily_mean'})
    # dt = dt.merge(daily_mean, on=['month', 'mday'], copy=False)
    #
    # del daily_mean
    #
    # wdaily_mean = dt[['week', 'wday', 'sales']].groupby(['week', 'wday'],  as_index=False)['sales'].mean()
    # wdaily_mean = wdaily_mean.rename(columns={'sales': 'wdaily_mean'})
    # dt = dt.merge(wdaily_mean, on=['week', 'wday'], copy=False)
    #
    # del wdaily_mean

    dt['freq'] = dt['freq']/dt['freq'].max()
    # dt['price'] = dt.groupby(['store_id', 'item_id', 'month'])['sell_price'].transform('mean')
    dt['price_momentum_m'] = dt['sell_price'] / dt.groupby(['id', 'month'])['sell_price'].transform('mean')
    # dt['price_momentum_q'] = dt['sell_price'] / dt.groupby(['id', 'quarter'])['sell_price'].transform('mean')
    #dt['price_momentum_y'] = dt['sell_price'] / dt.groupby(['id', 'year'])['sell_price'].transform('mean')

    # dt['zeros_28'] = dt['sales'].replace(0, np.nan)
    # dt['zeros_28'] = dt[["id", 'zeros_28']].groupby("id")['zeros_28'].transform(lambda x: x.rolling(28).count())

    # dt['zeros_56'] = dt['sales'].replace(0, np.nan)
    # dt['zeros_56'] = dt[["id", 'zeros_56']].groupby("id")['zeros_56'].transform(lambda x: x.rolling(112).count())

    return dt

df = df.drop(['E0', 'E2', 'E3', 'E5', 'E6',
            'E11', 'E12','E13',	'E14',
            'E15',	'E16',	'E17', 'E19',
            'E20',	'E21',	'E22',	'E23',
            'E24',	'E25', 'E28', 'E29',
            'E30', 'ET4'], axis=1)

df = create_lag_features(df)
df = create_date_features(df)

print(df.shape)
df_test = df[df.date >= (fday - timedelta(days=max_lags))]

# df_test = df_test.drop([
#                         'E0', 'E2', 'E3', 'E5', 'E6',
#                         'E11', 'E12','E13',	'E14',
#                         'E15',	'E16',	'E17', 'E19',
#                         'E20',	'E21',	'E22',	'E23',
#                         'E24',	'E25', 'E28', 'E29',
#                         'E30', 'ET4'], axis=1)

joblib.dump(df_test, 'test/test_nn.joblib')
#df_test.to_csv('ts.csv', index=False)

del df_test
gc.collect()
print(gc.collect())

drop_set = {}
for c in df.columns:
    ind_na = df[df[c].isna()].index
    num_na = len (ind_na)
    if num_na>0:
        print (c, num_na)
        drop_set = {*drop_set, *ind_na}
print (len(drop_set))
df_train = df.drop(drop_set)
print(df_train.shape)

del df
gc.collect()
print(gc.collect())

if fix_xmas:
    drop_index = df_train[df_train.d.isin(xmas_days)].index
    df_train = df_train.drop(drop_index)
print(df_train.shape)

#df_train = df_train.loc[df_train['sum_sales']>0, :]

# df_train = df_train.drop([
#                         'E0', 'E2', 'E3', 'E5', 'E6',
#                         'E11', 'E12','E13',	'E14',
#                         'E15',	'E16',	'E17', 'E19',
#                         'E20',	'E21',	'E22',	'E23',
#                         'E24',	'E25', 'E28', 'E29',
#                         'E30', 'ET4'], axis=1)

print(df_train.shape)
joblib.dump(df_train, 'test/train_nn.joblib')
#df_train.to_csv('tr.csv', index=False)
