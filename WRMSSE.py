from typing import Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm
import joblib

class WRMSSEEvaluator(object):

    group_ids = ('all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
                 ['state_id', 'cat_id'], ['state_id', 'dept_id'], ['store_id', 'cat_id'],
                 ['store_id', 'dept_id'], ['item_id', 'state_id'], ['item_id', 'store_id'])

    def __init__(self,
                 train_df: pd.DataFrame,
                 valid_df: pd.DataFrame,
                 calendar: pd.DataFrame,
                 prices: pd.DataFrame):
        '''
        intialize and calculate weights
        '''
        self.calendar = calendar
        self.prices = prices
        self.train_df = train_df
        self.valid_df = valid_df
        self.train_target_columns = [i for i in self.train_df.columns if i.startswith('d_')]
        self.weight_columns = self.train_df.iloc[:, -28:].columns.tolist()

        self.train_df['all_id'] = "all"

        self.id_columns = [i for i in self.train_df.columns if not i.startswith('d_')]
        self.valid_target_columns = [i for i in self.valid_df.columns if i.startswith('d_')]

        if not all([c in self.valid_df.columns for c in self.id_columns]):
            self.valid_df = pd.concat([self.train_df[self.id_columns], self.valid_df],
                                      axis=1,
                                      sort=False)
        self.train_series = self.trans_30490_to_42840(self.train_df,
                                                      self.train_target_columns,
                                                      self.group_ids)
        self.valid_series = self.trans_30490_to_42840(self.valid_df,
                                                      self.valid_target_columns,
                                                      self.group_ids)

        self.weights = self.get_weight_df()
        self.scale = self.get_scale()
        self.train_series = None
        self.train_df = None
        self.prices = None
        self.calendar = None

    def get_scale(self):
        '''
        scaling factor for each series ignoring starting zeros
        '''
        scales = []
        for i in tqdm(range(len(self.train_series))):
            series = self.train_series.iloc[i].values
            series = series[np.argmax(series != 0):]
            scale = ((series[1:] - series[:-1]) ** 2).mean()
            scales.append(scale)
        return np.array(scales)

    def get_name(self, i):
        '''
        convert a str or list of strings to unique string 
        used for naming each of 42840 series
        '''
        if type(i) == str or type(i) == int:
            return str(i)
        else:
            return "--".join(i)

    def get_weight_df(self) -> pd.DataFrame:
        """
        returns weights for each of 42840 series in a dataFrame
        """
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        weight_df = self.train_df[["item_id", "store_id"] + self.weight_columns].set_index(
            ["item_id", "store_id"]
        )
        weight_df = (
            weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"})
        )
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        weight_df = weight_df.merge(
            self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)[
            "value"
        ]
        weight_df = weight_df.loc[
                    zip(self.train_df.item_id, self.train_df.store_id), :
                    ].reset_index(drop=True)
        weight_df = pd.concat(
            [self.train_df[self.id_columns], weight_df], axis=1, sort=False
        )
        weights_map = {}
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False)):
            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis=1)
            lv_weight = lv_weight / lv_weight.sum()
            for i in range(len(lv_weight)):
                weights_map[self.get_name(lv_weight.index[i])] = np.array(
                    [lv_weight.iloc[i]]
                )
        weights = pd.DataFrame(weights_map).T / len(self.group_ids)

        return weights

    def trans_30490_to_42840(self, df, cols, group_ids, dis=False):
        '''
        transform 30490 sries to all 42840 series
        '''
        series_map = {}
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False, disable=dis)):
            tr = df.groupby(group_id)[cols].sum()
            for i in range(len(tr)):
                series_map[self.get_name(tr.index[i])] = tr.iloc[i].values
        return pd.DataFrame(series_map).T

    def get_rmsse(self, valid_preds) -> pd.Series:
        '''
        returns rmsse scores for all 42840 series
        '''
        score = ((self.valid_series - valid_preds) ** 2).mean(axis=1)
        rmsse = (score / self.scale).map(np.sqrt)
        return rmsse

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds],
                                axis=1,
                                sort=False)
        valid_preds = self.trans_30490_to_42840(valid_preds,
                                                self.valid_target_columns,
                                                self.group_ids,
                                                True)
        self.rmsse = self.get_rmsse(valid_preds)
        self.contributors = pd.concat([self.weights, self.rmsse],
                                      axis=1,
                                      sort=False).prod(axis=1)
        #self.weights.to_csv('weight.csv')

        return np.sum(self.contributors)


# train_df = pd.read_csv('data/sales_train_evaluation.csv')
# calendar = pd.read_csv('data/calendar.csv')
# prices = pd.read_csv('data/sell_prices.csv')
#
# train_fold_df = train_df.iloc[:, :-28]
#
# valid_fold_df = train_df.iloc[:, -28:].copy()
#
# e = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
# # joblib.dump(e, 'eval.joblib')
# del train_fold_df, train_df, calendar, prices
#
# sub = pd.read_csv('submission_3.csv')
# print(e.score(np.asarray(sub.iloc[:30490, 1:])))
#
e = joblib.load('eval.joblib')
# pred = joblib.load('test_pred.joblib')

# w = pd.read_csv('weight.csv')
# zeros = w.loc[w['weight_v']==0, 'id']


#sub = pd.read_csv('submission_%s.csv' %i)


for epoch in [1000]:
    sub1 = pd.read_csv('submission/sub_future_%s_%s.csv' % (12, 1))
    sub1 = sub1.set_index('id', drop=True)

    sub2 = pd.read_csv('submission/sub_future_%s_%s.csv' % (10, 1))
    sub2 = sub2.set_index('id', drop=True)

    sub3 = pd.read_csv('submission/sub_future_%s_%s.csv' % (1450, 1))
    sub3 = sub3.set_index('id', drop=True)

    sub1.iloc[30490:, -28:] = np.asarray(sub1.iloc[:30490, -28:])
    sub2.iloc[30490:, -28:] = np.asarray(sub2.iloc[:30490, -28:])

    print('Sub1', e.score(np.asarray(sub1.iloc[30490:, -28:])))
    print('Sub2', e.score(np.asarray(sub2.iloc[30490:, -28:])))
    print('Sub3', e.score(np.asarray(sub3.iloc[30490:, -28:])))

    print('Average', e.score((np.asarray(sub2.iloc[30490:, :])*0.5 + np.asarray(sub3.iloc[30490:, :])*0.5)))

    sub2.to_csv('sub_nn.csv')

    sub1.iloc[30490:, :] = (np.asarray(sub2.iloc[30490:, :])*0.5 + np.asarray(sub3.iloc[30490:, :])*0.5)
    sales = pd.read_csv('data/sales_train_evaluation.csv')

    sub1.iloc[:30490, :] = np.asarray(sales.iloc[:, -28:])
    print(sales.iloc[:, -28:].shape)
    max_values = np.max(np.asarray(sales.iloc[:, -28:]), axis=1)

    for day in range(28):
        sub1.iloc[30490:, day] = np.where(max_values == 0, 0, np.asarray(sub1.iloc[30490:, day]))

    label = pd.read_csv('label.csv')
    label = pd.pivot_table(label, values='sales', index=['id'], columns=['d'])

    print(label.head())

    for day in range(28):
        sub1.iloc[30490:, day] = np.where(label.iloc[:30490, day] < 0.1, 0, np.asarray(sub1.iloc[30490:, day]))
        sub1.iloc[30490:, day] = np.where(sub1.iloc[30490:, day] < 0.1, 0, np.asarray(sub1.iloc[30490:, day]))
        sub1.iloc[30490:, day] = np.where(sub1.iloc[30490:, day] < 0, 0, np.asarray(sub1.iloc[30490:, day]))

    print('Sub1', e.score(np.asarray(sub1.iloc[:30490, -28:])))
    print('Sub1', e.score(np.asarray(sub1.iloc[30490:, -28:])))

    sub1.to_csv('final_sub.csv')

