import pandas as pd
import numpy as np
import re as re
import feather as ftr
from base import Feature, get_arguments, generate_features
from k_means import KMeansFeaturizer
from sklearn.preprocessing import MinMaxScaler
from __init__ import reduce_mem_usage


Feature.dir = 'features'

class Outliers(Feature):
    def create_features(self):
        global train, test
        data = train.append(test)
        data['outliers'] = 0
        
        data.loc[data.query('0.463<var_108<0.465').index.values, 'outliers'] = 1
        data.loc[data.query('0.0976<var_12<0.0985').index.values,'outliers'] = 1
        data.loc[data.query('0.0633<var_126<0.0637').index.values, 'outliers'] = 1
        data.loc[data.query('0.059<var_181<0.061').index.values, 'outliers'] = 1
        
        test = data.iloc[train.shape[0]:, :].reset_index(drop=True)
        train = data.iloc[:train.shape[0], :].reset_index(drop=True)

        self.train['outliers'] = train['outliers']
        self.test['outliers'] = test['outliers']

class Spike_var_108(Feature):
    def create_features(self):
        global train, test
        data = train.append(test)

        data['spike_var_108'] = 0
        spike108_index = data.query('0.463<var_108<0.465').index.values
        data.loc[spike108_index, 'spike_var_108'] = 1

        self.train['spike_var_108'] = train['spike_var_108']
        self.test['spike_var_108'] = test['spike_var_108']

class Base(Feature):
    def create_features(self):
        for i in range(200):
            self.train[f'var_{i}'] = train[f'var_{i}']
            self.test[f'var_{i}'] = test[f'var_{i}']


class K_means(Feature):
    """
    スケーリングしてからクラスタリング、出力はone-hotベクトルなのでスケーリングが関係なくなる
    """
    def create_features(self):
        global train, test

        data = train.append(test)
        data.drop(columns=['target', 'ID_code'], inplace=True)

        kmf = KMeansFeaturizer()
        k_data = kmf.fit_transform(data)

        data = pd.DataFrame(k_data.toarray())
        data = data.add_prefix('k_')

        test = data.iloc[train.shape[0]:, :].reset_index(drop=True)
        train = data.iloc[:train.shape[0], :].reset_index(drop=True)

        for i in range(100):
            self.train[f'k_{i}'] = train[f'k_{i}']
            self.test[f'k_{i}'] = test[f'k_{i}']


class BigDiffMean(Feature):
    '''
    平均値の差が大きいものを選出している。
    スケールしてから選ぶ。
    '''
    def create_features(self):
        global train, test

        features = train.columns.values[2:202]
        t0 = train.loc[train['target'] == 0]
        t1 = train.loc[train['target'] == 1]
        train_ID_df = train.iloc[:, :2]
        test_ID_df = test.iloc[:, :1]

        train.drop(columns=['target', 'ID_code'], inplace=True)
        test.drop(columns=['ID_code'], inplace=True)

        plus_idxs = (t1[features].mean(axis=0) - t0[features].mean(axis=0)
                    ).sort_values(ascending=False)[0:10].index

        minus_idxs = (t1[features].mean(axis=0) - t0[features].mean(axis=0)
                    ).sort_values(ascending=False)[-10:].index

        self.train['plus_diff_mean'] = train.loc[:,plus_idxs].sum(axis=1)
        self.test['plus_diff_mean'] = test.loc[:,plus_idxs].sum(axis=1)

        self.train['minus_diff_mean'] = train.loc[:,minus_idxs].sum(axis=1)
        self.test['minus_diff_mean'] = test.loc[:,minus_idxs].sum(axis=1)


if __name__ == '__main__':

    args = get_arguments()

    train = ftr.read_dataframe('./data/input/min_maxed_train.feather')
    test = ftr.read_dataframe('./data/input/min_maxed_test.feather')

    train = reduce_mem_usage(train, verbose=True)
    test = reduce_mem_usage(test, verbose=True)
    
    generate_features(globals(), args.force)
