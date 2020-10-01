import pandas as pd
import random

class KFoldValidation():
    df = None
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    def train_with_kfold(self, options):
        algorithm = options['train_algorithm']
        self.df = options['df']
        num_folds = options['num_folds']
        label_column = options['label_column']
        beta = options['beta']
        positive_class = options['positive_class']

        model = algorithm.train(options)

        acc_list = []
        f_sc_list = []

        folds = self._split_in_k_folds(num_folds, label_column)
        for index, _ in enumerate(folds):
            options['train_dataset'] = pd.concat(folds[0:index]+folds[index+1:]) # train is all but the test concatenated
            test_set = folds[index]
            for _, row in test_set.iterrows():
                correct = row[label_column] # the predict function stores in the confusion matrix
                predicted = model.predict(row.drop(label_column), options) # predict for each row
                self._evaluate_to_confusion_matrix(correct, predicted, positive_class)
            acc_list.append(self._get_accuracy())
            f_sc_list.append(self._get_f_score(beta))
        
        ac_metric_mean, ac_metric_std = self._get_statistics(acc_list)
        f_metric_mean, f_metric_std = self._get_statistics(f_sc_list)
        print(f"Median,{ac_metric_mean},{f_metric_mean}")
        print(f"StandardDeviation,{ac_metric_std},{f_metric_std}")

    def _split_in_k_folds(self, num_folds, label_column):
        """
        n_folds: integer, number of folds to split the data
        label_column: string, column target in the prediction, used to split the data
        return: list of DataFrames, one for each fold

        Stratified kfold using the label_column as class
        """
        groups = self.df.groupby([label_column]).groups #group elements by its class
        classes = groups.keys()
        folds = [[] for k in range(num_folds)]
        groups_index_list = []

        for i in classes: #for each class, get the list of indexes in the original dataframe
            groups_index_list.append(groups[i].tolist())

        for i in groups_index_list:
            random.shuffle(i) # shuffle the list of index so it's randomly splitted 
            range_in_fold = round(len(i)/num_folds)
            current = 0
            for j in range(num_folds):
                folds[j] += i[current:( (j+1)*range_in_fold)] # add this list with indexes to the fold
                current = (j+1)*range_in_fold
        
        return [self.df.iloc[i] for i in folds] # return a list with dataframes for each fold

    def _get_statistics(self, population):
        population_sum = sum(population)
        population_size = len(population)

        mean = population_sum/population_size

        result = 0
        for i in population:
            result += (i-mean)**2
        result = result/population_size
        standard_deviation = result**1/2

        return mean, standard_deviation


    def _evaluate_to_confusion_matrix(self, correct_class, predicted_class, positive_class):
        if predicted_class == correct_class:
            if predicted_class == positive_class:
                self.true_positive += 1
            else:
                self.true_negative += 1
        else:
            if predicted_class == positive_class:
                self.false_positive += 1
            else:
                self.false_negative += 1

    def _get_accuracy(self):
        return (self.true_positive + self.true_negative) / (self.true_negative + self.false_negative + self.true_positive + self.false_positive)

    def _get_recall(self):
        return self.true_positive / (self.true_positive + self.false_negative)

    def _get_precision(self):
        return self.true_positive / (self.true_positive + self.false_positive)

    def _get_f_score(self, beta):
        base_value = 1+(beta ** 2)
        numerator = self._get_precision() * self._get_recall() 
        denominator = ( (beta**2) * self._get_precision() ) + self._get_recall()
        return base_value * (numerator/denominator)
