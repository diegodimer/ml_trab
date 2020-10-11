from utils import get_bootstrap, get_test_set_from_bootstrap, normalize_df
import pandas as pd
from decisionTree import DecisionTree
from base import BaseAlgorithm

class RandomForest(BaseAlgorithm): #to-do: add base

    ensemble = []

    def train(self, options):
        """
        train a random forest, using n_trees decision trees
        options['df']: pandas dataframe
        options['n_trees']: number of trees
        options['label_column']: label column to be predicted
        options['bootstrap_size']: the size of the bootstrap, entries not used in the bootstrap will be ignored
        """
        num_trees = options['n_trees']
        df = options['df']
        bootstrap_size = options['bootstrap_size']

        tree_options = {
            'label_column': options['label_column']
        }
        for i in range(num_trees):
            tree_options.update({
                'df': get_bootstrap(df, bootstrap_size)
            })
            new_tree = DecisionTree()
            self.ensemble.append(new_tree.train(options))

        return self

    def predict(self, inference_data):
        predictions = {}
        for tree in self.ensemble:
            predicted = tree.predict(inference_data)
            if predicted in predictions:
                predictions[predicted] += 1
            else:
                predictions[predicted] = 1

        return max(predictions, key=lambda k: predictions[k])