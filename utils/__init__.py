import time
import pandas as pd
from contextlib import contextmanager
import feather as ftr

@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_datasets(feats):
    dfs = [ftr.read_dataframe(f'features/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [ftr.read_dataframe(f'features/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    features=X_train.columns.values
    return X_train, X_test,features


def load_target(target_name):
    train = pd.read_csv('./data/input/train.csv')
    y_train = train[target_name]
    return y_train
