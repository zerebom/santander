from scripts.data_augumation import augmation
from sklearn.model_selection import KFold, StratifiedKFold
from models.lgbm import train_and_predict
from logs.logger import log_best
from utils import load_datasets, load_target
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
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
logging.debug(feats)

target_name = config['target_name']

X_train_all, X_test, features = load_datasets(feats)

# reshape to be [samples][pixels][width][height]
featureLayer1 = [Conv2D(64, (3, 3), padding='same', input_shape=X_train_all.shape[1:]),
                 Activation('relu'),
                 Conv2D(64, (3, 3), padding='same'),
                 Activation('relu'),
                 MaxPooling2D(pool_size=(2, 2)),
                 Dropout(0.25)]

featureLayer2 = [Conv2D(128, (3, 3), padding='same'),
                 Activation('relu'),
                 Conv2D(128, (3, 3), padding='same'),
                 Activation('relu'),
                 MaxPooling2D(pool_size=(2, 2)),
                 Dropout(0.25)]

featureLayer3 = [Conv2D(256, (3, 3), padding='same'),
                 Activation('relu'),
                 Conv2D(256, (3, 3), padding='same'),
                 Activation('relu'),
                 Conv2D(256, (3, 3), padding='same'),
                 Activation('relu'),
                 MaxPooling2D(pool_size=(2, 2)),
                 Dropout(0.25)]

fullConnLayer = [Flatten(),
                 Dense(1024),
                 Activation('relu'),
                 Dropout(0.5),
                 Dense(1024),
                 Activation('relu'),
                 Dropout(0.5)]

classificationLayer = [Dense(num_classes),
                       Activation('softmax')]

model = Sequential(featureLayer1 + featureLayer2 +
                   featureLayer3 + fullConnLayer + classificationLayer)

model.compile(loss='mean_squared_error',
              optimizer=RMSprop(), metrics=['accuracy'])
