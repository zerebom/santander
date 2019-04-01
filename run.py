import pandas as pd
import datetime
import logging
from sklearn.model_selection import KFold
import argparse
import json
import numpy as np

from utils import load_datasets, load_target
from logs.logger import log_best
#lgbm関係はmodel.pyからimportしている
from models.lgbm import train_and_predict


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now))

#このfeatsは特徴量ファイル名(カラム名ではないので、複数列でも可(base等))
feats = config['features']
logging.debug(feats)

target_name = config['target_name']

X_train_all, X_test = load_datasets(feats)
y_train_all = load_target(target_name)
logging.debug(X_train_all.shape)

y_preds = []
models = []

lgbm_params = config['lgbm_params']

kf = KFold(n_splits=3, random_state=0)
for train_index, valid_index in kf.split(X_train_all):
    X_train, X_valid = (
        X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :]
    )
    y_train, y_valid = y_train_all[train_index], y_train_all[valid_index]

    # lgbmの実行
    y_pred, model = train_and_predict(
        X_train, X_valid, y_train, y_valid, X_test, lgbm_params
    )

    # 結果の保存
    y_preds.append(y_pred)
    models.append(model)

    # スコア(logger.pyで定義)
    log_best(model, config['loss'])

# CVスコア
scores = [
    m.best_score['valid_0'][config['loss']] for m in models
]
score = sum(scores) / len(scores)
print('===CV scores===')
print(scores)
print(score)
logging.debug('===CV scores===')
logging.debug(scores)
logging.debug(score)

# submitファイルの作成
ID_name = config['ID_name']
sub = pd.DataFrame(pd.read_csv('./data/input/test.csv')[ID_name])

print(y_preds)

y_sub = sum(y_preds) / len(y_preds)

print(y_sub)
# if y_sub.shape[1] > 1:
#     y_sub = np.argmax(y_sub, axis=1)

sub[target_name] = y_sub

sub.to_csv(
    './data/output/sub_{0:%Y%m%d%H%M%S}_{1}.csv'.format(now, score),
    index=False
)
