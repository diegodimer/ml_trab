import pandas as pd
import random
class KFoldValidation():

    def train_with_kfold(self, options):
        """
        Training using kfoldvalidation
        options['df']: pandasDataFrame, the training dataframe
        options['train_algorithm']: the training algorithm (class)
        options['num_folds']: number of folds
        + all options necessary for the training algorithm
        """
        algorithm = options['train_algorithm']
        df = options['df']
        num_folds = options['num_folds']
        label_column = options['label_column']

        acc_list = []

        folds = self._split_in_k_folds(num_folds, label_column, df)
        print("index,score,test_fold_size,accuracy")
        for index, _ in enumerate(folds):

            options['df'] = pd.concat(folds[0:index]+folds[index+1:]) # train is all but the test concatenated
            model = algorithm.train(options)
            test_set = folds[index]
            test_set_size = len(test_set.index)

            score = 0
            for _, row in test_set.iterrows():
                correct = row[label_column]
                predicted = model.predict(row.drop(label_column)) # predict for each row
                if predicted == correct:
                    score += 1
            accuracy = score/test_set_size
            print(f"{index},{score},{test_set_size},{accuracy}")
            acc_list.append(accuracy)

        return sum(acc_list)/len(acc_list)

    def _split_in_k_folds(self, num_folds, label_column, df):
        """
        n_folds: integer, number of folds to split the data
        label_column: string, column target in the prediction, used to split the data
        return: list of DataFrames, one for each fold

        Stratified kfold using the label_column as class
        """
        groups = df.groupby([label_column]).groups #group elements by its class
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
        
        return [df.iloc[i] for i in folds] # return a list with dataframes for each fold
