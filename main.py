from KFoldValidation import KFoldValidation
from randomForest import RandomForest
from decisionTree import DecisionTree
import pandas as pd
from math import sqrt

df = pd.read_csv('datasets/wine-recognition.tsv', sep='\t')

options = {
    'train_algorithm' : RandomForest(),
    'df' : df,
    'label_column' : 'target',
    'num_folds' : 10,
    'n_trees': 10,
    'bootstrap_size': 10
    }
runner = KFoldValidation()
runner.train_with_kfold(options)
