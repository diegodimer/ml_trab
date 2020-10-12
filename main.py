from KFoldValidation import KFoldValidation
from randomForest import RandomForest
import pandas as pd
import random
import numpy as np

seed = 5
np.random.seed(seed)
random.seed(seed)

df = pd.read_csv('datasets/iris.data')

options = {
    'train_algorithm' : RandomForest(),
    'df' : df,
    'label_column' : 'Y',
    'num_folds' : 5,
    'n_trees': 15,
    'bootstrap_size': 2,
    }

runner = KFoldValidation()
runner.train_with_kfold(options)