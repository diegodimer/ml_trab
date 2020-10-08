import math
import random
import pandas as pd

class DecisionTree():

    true_positive = true_negative = false_positive = false_negative = 0


    '''
        df: training dataFrame
        outcome: name of attribute we want to predict
    '''
    def train(self, options):
        df = options['df']
        self.outcome = options['label_column']

        allAtt = list(df.columns) # get list of attributes
        allAtt.remove(outcome)

        self.root = self._makeTree(df, "root", allAtt) # build decision tree

        return root


    '''
        inference_data = pandas.core.series.Series, should be the row (without the label) to be predicted by the algorithm
    '''
    def predict(self, inference_data):

        node = self.root

        while True: # traverse the tree until we find a leaf node
            if len(node.children) == 0:
                break
            else:
                v = str(inference_data[node.attribute])
                for child in node.children:
                    if child.parentValue == v:
                        node = child
                        break

        return node.attribute
        

    '''
        implementação de uma árvore de decisão (slide 43 da aula 3)

        recursive
    '''
    def _makeTree(self, df, parentValue, listAtt):
        # if all rows have the same outcome or there's no attributes left, create leaf node
        if df[self.outcome].nunique() == 1 or len(listAtt) == 0: 
            return Node(str(df[self.outcome].mode()[0]), parentValue)

        mAtt = random.sample(listAtt, math.ceil(math.sqrt(len(listAtt)))) # select m random attributes
   
        newAttribute = self._InfoGain(df, mAtt) # selects best attribute
       
        listAtt.remove(newAttribute) # removes selected attribute from list

        newNode = Node(newAttribute, parentValue) #creates new node

        groups = df.groupby(df[newAttribute]) # splits data according to the selected attribute values 
        for name, obj in groups:
            newNode.addChildren(self._makeTree(obj, str(name), listAtt.copy())) # adds one children for each value

        return newNode

    '''
        implementação de Ganho de Informação (slides 47-53 aula 3)
    '''
    def _InfoGain(self, df, listAtt):
        info_D = self._Entropy(df)

        best = 0
        for attr in listAtt:
            info_aD = 0
            for value in df[attr].unique():
                dj = df.loc[df[attr] == value]
                info_aD += len(dj)/len(df)*self._Entropy(dj)

            gain = info_D - info_aD
            
            if gain >= best:
                best = gain
                chosen = attr
        return chosen


    # auxiliar para _InfoGain
    def _Entropy(self, df):

        info_D = 0
        for yClass in df[self.outcome].unique():
            p_i = len(df.loc[ df[self.outcome] == yClass ])/len(df)
            info_D += - p_i*math.log(p_i,2)

        return info_D



'''
    Node from a decision tree

    attribute: the attribute (also called column, X_i, feature) being considered in this node. 
        If it is a leaf node, this variable stores the final prediction
    
    parentValue: the VALUE of the parent attribute (each parent has one children for each possible value of its attribute)

    children: a list of nodes
'''
class Node():

    def __init__(self, attribute, parentValue):
        self.attribute = attribute
        self.parentValue = parentValue
        self.children = []

    def isLeaf(self):
        return True if len(self.children) == 0 else False

    def addChildren(self, obj):
        self.children.append(obj)
    



# just for testing
def print_tree(node, tb):
    tab = '----'*tb
    print(tab+node.attribute+' '+node.parentValue+' '+str(len(node.children))+' ')
    
    for child in node.children:
        print_tree(child,tb+1)

# just for testing
if __name__ == '__main__':

    '''
    dataframe = pd.read_csv('testdata.csv')
    dt = DecisionTree()
    print(dt._InfoGain(dataframe,['age','income','credit_rating']))
    '''
    dataframe = pd.read_csv('benchmark.csv', sep = ';')

    dt = DecisionTree()
    dt.train(dataframe, 'Joga')

    print_tree(dt.root,0)

    series = dataframe.loc[3]

    print(dt.predict(series))
