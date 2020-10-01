from base import BaseAlgorithm

class KNNAlgorithm(BaseAlgorithm):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    def train(self, options):
        """
        Knn Algorithm doesn't train a model. It calculates directly on the training data the 
        label for some entry.
        """   
        return self

    def predict(self, inference_data, options):
        """
        inference_data = pandas.core.series.Series, should be the row (without the label) to be predicted by the algorithm
        options = dictionary, contains the inference options, for Knn algorithm it's necessary to have:
            * df = the dataframe used for the inference,
            * label_column = the name of the column to be predicted
            * distance_function = the function used to calculate the distance (euclidean or manhattan)
            * number_of_neighbors = number of n neighbors to calculate the label
        """

        df = options['df']
        label_column = options['label_column']
        distance_function = options['distance_function']
        number_of_neighbors = options['number_of_neighbors']

        distances_list = []

        for index, row in df.iterrows():
            distance = distance_function(row.drop(label_column), inference_data)
            distances_list.append( (distance, row[label_column]))
        distances_list = sorted(distances_list)
        distances_list = distances_list[0:number_of_neighbors] # get the N nearest neighbots

        predicted_class = self._get_most_frequent(distances_list) # calculate the most frequent neighbor

        if 'add_to_confusion_matrix' in options and options['add_to_confusion_matrix'] == True:
            self._evaluate_to_confusion_matrix(options, predicted_class)

        return predicted_class

    def get_accuracy(self):
        return (self.true_positive + self.true_negative) / (self.true_negative + self.false_negative + self.true_positive + self.false_positive)

    def get_recall(self):
        return self.true_positive / (self.true_positive + self.false_negative)

    def get_precision(self):
        return self.true_positive / (self.true_positive + self.false_positive)

    def get_f_score(self, beta):
        base_value = 1+(beta ** 2)
        numerator = self.get_precision() * self.get_recall() 
        denominator = ( (beta**2) * self.get_precision() ) + self.get_recall()
        return base_value * (numerator/denominator)

    def _evaluate_to_confusion_matrix(self, options, predicted_class):
        if predicted_class == options['correct_class']:
            if predicted_class == options['positive_class']:
                self.true_positive += 1
            else:
                self.true_negative += 1
        else:
            if predicted_class == options['positive_class']:
                self.false_positive += 1
            else:
                self.false_negative += 1

    def _get_most_frequent(self, list_of_distances):
        counter = {}
        for distance, label in list_of_distances:
            if label in counter:
                counter[label] += 1
            else:
                counter[label] = 1

        return max(counter.keys(), key=lambda key: counter[key])

    def print_results(self, index, beta):
        if index==0:
            print("FoldTest/Iter,Accuracy,F-1Measure")
        accuracy = self.get_accuracy()
        f_score = self.get_f_score(beta)
        print(f"{index},{accuracy},{f_score}")
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0
        self.true_negative = 0
        return accuracy, f_score
