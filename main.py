from KFoldValidation import KFoldValidation
from decisionTree import DecisionTree
import pandas as pd
from math import sqrt

df = pd.read_csv('iris.data')

options = {
    'train_algorithm' : DecisionTree(),
    'df' : df,
    'label_column' : 'Y',
    'num_folds' : 5,
    'n_trees': 100,
    'bootstrap_size': 100
    }
runner = KFoldValidation()
runner.train_with_kfold(options)