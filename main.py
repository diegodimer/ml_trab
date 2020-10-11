from KFoldValidation import KFoldValidation
from randomForest import RandomForest
from decisionTree import DecisionTree
import pandas as pd
from math import sqrt
from copy import deepcopy

df = pd.read_csv('datasets/wine-recognition.tsv', sep='\t')
original_df = deepcopy(df)
options = {
    'train_algorithm' : RandomForest(),
    'df' : df,
    'label_column' : 'target',
    'num_folds' : 5,
    'n_trees': 10,
    'bootstrap_size': 2,
    }
runner = KFoldValidation()
# runner.train_with_kfold(options)

l = [1, 5, 10, 15, 25, 50]
acc = []
for i in l:
    options.update({"n_trees": i, "df": df})
    acc.append(runner.train_with_kfold(options))

import matplotlib.pyplot as plt
plt.plot(l, acc)
plt.ylabel("Acurácia média")
plt.xlabel("n_trees")
plt.savefig('n_trees.png')