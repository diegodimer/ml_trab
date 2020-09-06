from KFoldValidation import KFoldValidation
from KNNAlgorithm import KNNAlgorithm
import pandas as pd
from math import sqrt

def euclidean_distance(a, b):
    assert a.size == b.size, "Rows should have the same number of columns"
    acc = 0
    for name, value in a.iteritems():
        acc += (value - b[name]) ** 2
    return sqrt(acc)

def normalize_df(self):
    """
    using the formula
    element - min   element / max element - min element
    """
    return (df-df.min()) / (df.max()-df.min())


df = pd.read_csv('diabetes.csv')
df = normalize_df(df)

options = {
    'train_algorithm' : KNNAlgorithm(),
    'train_dataset' : df,
    'label_column' : 'Outcome',
    'distance_function' : euclidean_distance,
    'number_of_neighbors' : 10,
    'add_to_confusion_matrix' : True,
    'num_folds' : 10,
    'positive_class': 1.0,
    'negative_class': 0.0
    }
runner = KFoldValidation()
runner.train_with_kfold(options)