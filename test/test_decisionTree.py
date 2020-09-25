from decisionTree import DecisionTree
import pandas as pd
import unittest

class TestDecisionTree(unittest.TestCase):
    def test_benchmark(self):
        options = {
            'df': pd.read_csv("benchmark.csv", sep=';'),
            'label_column': "Joga"
        }
        tr = DecisionTree()
        model = tr.train(options)

        inf_data = pd.Series(["Ensolarado", "Quente", "Normal", "Verdadeiro"], index=["Tempo", "Temperatura", "Umidade", "Ventoso"], name ="InferenceData")
        self.assertEqual(model.predict(inf_data), 'Sim')
    
    def test_all_file(self):
        options = {
            'df': pd.read_csv("benchmark.csv", sep=';'),
            'label_column': "Joga"
        }
        tr = DecisionTree()
        model = tr.train(options)

        for _, row in options['df'].iterrows():
            target_label = row["Joga"]
            predicted = model.predict(row.drop("Joga"))
            self.assertEqual(target_label, predicted)
