from base import BaseAlgorithm
from collections import namedtuple
import pandas as pd
import math
import random

Node = namedtuple('Node', ['label', 'children'])

class DecisionTree(BaseAlgorithm):

    def _select_attribute(self, attributes_list, df):
        """
        Select the attribute to split the decision tree based on the concept of entropy
        df: the pandas dataframe 
        """
        entropy_all_data = self._entropy(df)

        mAtt = random.sample(attributes_list, math.ceil(math.sqrt(len(attributes_list)))) # select m random attributes
       
        best = 0
        
        for attr in mAtt:
            entropy_attribute = 0
            for value in df[attr].unique():
                df_attribute = df.loc[df[attr] == value]
                entropy_attribute += len(df_attribute)/len(df)*self._entropy(df_attribute)

            gain = entropy_all_data - entropy_attribute

            if gain >= best:
                best = gain
                chosen = attr
        return chosen


    # auxiliar para _InfoGain
    def _entropy(self, df):
        entropy = 0
        for attr_class in df[self.outcome].unique():
            p_i = len(df.loc[ df[self.outcome] == attr_class ])/len(df)
            entropy += - p_i*math.log(p_i,2)
        return entropy

        
    def train(self, options):
        """
        Train a decision tree
        options['df']: dataframe
        options['label_colum']: name of the attribute to be predicted
        """
        key_column =  options['label_column'] # column with the class name
        self.outcome = key_column
        df = options['df']
        L = df.columns.to_list()

        L.remove(key_column)
        self.tree = self._recursive_tree_generator(options['df'], key_column, L)

        return self

    def _recursive_tree_generator(self, df, label_column, L):
        # df = dataframe
        # L = lista de atributos preditivos em D

        if len(df[label_column].unique()) == 1: # o nó é folha
            node = self._new_node(df[label_column].iloc[0]) # o label dele é o que tem aí
            return node

        if len(L) == 0:
            node = self._new_node(df[label_column].mode()[0]) # se L é vazia, a classe é a mais frequente no DF
            return node

        chosen_attr = self._select_attribute(L, df)
        node = self._new_node(chosen_attr)
        L.remove(chosen_attr)
        for distinc_value, new_df in df.groupby(df[chosen_attr]): #pra cada valor distinto do atributo escolhido
            if len(new_df.index) == 0:
                node = self._new_node(df[label_column].mode()[0])
                return node
            else:
                self._add_children(father = node,
                                child = self._recursive_tree_generator(new_df, label_column, L.copy()),
                                transition = distinc_value)
        return node

    def _new_node(self, label):
        return Node(label, [])
    
    def _add_children(self, father=None, child=None, transition=None):
        father.children.append( [child, transition] )

    def predict(self, inference_data):
        return self.recursive_tree_search(inference_data, self.tree)


    def recursive_tree_search(self, data, tree):
        if len(tree.children) == 0:
            return tree.label

        attribute = data[tree.label]

        for child in tree.children:
            if child[1] == attribute:
                return self.recursive_tree_search(data, child[0])
