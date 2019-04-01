import pandas as pd
import numpy as np
import re as re
import feather as ftr
from base import Feature, get_arguments, generate_features

Feature.dir = 'features'


class Base(Feature):
    def create_features(self):
        for i in range(200):
            self.train[f'var_{i}'] = train[f'var_{i}']    
            self.test[f'var_{i}'] = test[f'var_{i}']




if __name__ == '__main__':
    
    args = get_arguments()

    train = ftr.read_dataframe('./data/input/train.feather')
    test = ftr.read_dataframe('./data/input/test.feather')

    generate_features(globals(), args.force)
