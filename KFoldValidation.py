import pandas as pd
import random

class KFoldValidation():
    df = None

    def send_sample_inference(self, options):
        algorithm = options['train_algorithm']
        df = options['train_dataset']
        model = algorithm.train(df)

        column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        column_values = [1.0, 93.0, 70.0, 31.0, 0.0, 30.4, 0.315, 23.0]
        inference_data = pd.Series(column_values, index = column_names)
        predicted_class = model.predict(inference_data, options)
        print(predicted_class)

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

    def _normalize_df(self):
        """
        using the formula
        element - min   element / max element - min element
        """
        self.df = (self.df-self.df.min()) / (self.df.max()-self.df.min())

