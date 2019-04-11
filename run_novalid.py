from scripts.data_augumation import augmation
from sklearn.model_selection import KFold, StratifiedKFold
from models.lgbm import train_and_predict_novalid
from logs.logger import log_best
from utils import load_datasets, load_target
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import logging
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
import numpy as np
np.random.seed(seed=42)
# lgbm関係はmodel.pyからimportしている

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now))

# このfeatsは特徴量ファイル名(カラム名ではないので、複数列でも可(base等))
feats = config['features']
lgbm_params = config['lgbm_params']

logging.debug(feats)

target_name = config['target_name']

X_train_all, X_test, features = load_datasets(feats)
y_train_all = load_target(target_name)
logging.debug(X_train_all.shape)


# lgbmの実行
y_pred, model = train_and_predict_novalid(
    X_train_all, y_train_all, X_test, lgbm_params
)

# 結果の保存


ID_name = config['ID_name']
sub = pd.DataFrame(pd.read_csv('./data/input/test.csv')[ID_name])


sub[target_name] = y_pred

sub.to_csv(
    './data/output/final_sub_{0:%Y%m%d%H%M%S}_.csv'.format(now),
    index=False
)
