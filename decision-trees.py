

import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """

    gini = 0.0
    ##returning the last collumn of the dataset
    lastCollumn = data[:, -1]
    ##count the amount of appearences of each feature
    appearences = np.unique(lastCollumn, return_counts=True)[1]
    ##counting the amount of appearences with respect to the amount of features
    frequency = appearences / len(data)
    gini = 1 - np.sum(frequency ** 2)
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ##returning the last collumn of the dataset
    lastCollumn = data[:, -1]
    ##count the amount of appearences of each feature
    appearences = np.unique(lastCollumn, return_counts=True)[1]
    ##counting the amount of appearences with respect to the amount of features
    frequency = appearences / len(data)
    entropy = (-1) * np.sum(frequency * np.log2(frequency))

    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    
    
    def split(self):
        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to, and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ##variables to use later
        maximal_groups = []
        maximal_goodness = 0
        ## checks if we are a leaf
        if self.depth >= self.max_depth:
            self.terminal = True
            return


        for feature in range(self.data.shape[1] - 1):
            ##checking the goodness of split using the goodness of split function
            goodness = self.goodness_of_split(feature)[0]
            groups = self.goodness_of_split(feature)[1]
            if goodness > maximal_goodness:
                maximal_goodness = goodness
                maximal_groups = groups
                self.feature = feature
        ##checks if the group is a leaf
        if len(maximal_groups) <= 1:
            self.terminal = True
            return
         
        #create children
        for value, data in maximal_groups.items():
            child = DecisionNode(data, impurity_func=self.impurity_func, depth=self.depth+1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(child, value)
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        # declaring variables needed for calculations
        appearence = np.unique(self.data.T[-1], return_counts=True)[0]
        frequency = np.unique(self.data.T[-1], return_counts=True)[1]
        dict_param_count = {appearence[i]: frequency[i] for i in range(len(appearence))}

        # finding the predicition with the highest frequency
        pred = max(dict_param_count, key=dict_param_count.get)
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        total_impurity = self.impurity_func(self.data)
        sum_feature_importance = 0
    
        for child in self.children: ##iterates over all the children
            child_impurity = self.impurity_func(child.data) ##calculates the impurity of the child
            child_samples = len(child.data) ##calculates the amount of appearences
            child_impurity_weighted = child_samples / n_total_sample * child_impurity 
            sum_feature_importance += (total_impurity - child_impurity_weighted) ##sums up the reduction of the impuriy with the weighted amount
    
        self.feature_importance = sum_feature_importance / n_total_sample
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.
        Optionally calculates the Gain Ratio if self.gain_ratio is True.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split or gain ratio
        - groups: a dictionary holding the data after splitting 
                according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        # declaring the imurity function based on self
        imp_Funct = self.impurity_func

        # declaring variables for later use
        var = np.unique(self.data.T[feature])
        attribute = 0
        database = imp_Funct(self.data)
        splitinformation = 0

        # calculating goodnesss of split based on recitation formula
        for val in var:
            groups[val] = self.data[self.data[:, feature] == val]
            appearence_attribute = len(groups[val]) / len(self.data)
            splitinformation += appearence_attribute * np.log2(appearence_attribute)
            attribute += appearence_attribute * imp_Funct(groups[val])

    # if gain ratio is true then we need to divide by splitinformation
        if self.gain_ratio == True:
            information_gain = database - attribute

        # To avoid division by 0.
            if splitinformation == 0:
                return 0, groups

            # Adjust the variable according to the formula.
            splitinformation = splitinformation * (-1)
            goodness = information_gain / splitinformation

    # else if its false, we just reduce
        else:
            goodness = database - attribute
        return goodness, groups
    



def depth_of_node(node):
        # check if we are working with a leaf
        if node.terminal:
            return node.depth

        # Creating a list that will contain the depth of the node's children
        depths = []

        # Going over all the children of the node.
        for child in node.children:
            child_depth = depth_of_node(child)
            depths.append(child_depth)
        return max(depths)
    
                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        
            # Construct the root node and a queue to hold all the nodes
        self.root = DecisionNode(self.data, self.impurity_func, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        queue = [self.root]

        while queue:
        # Remove the current node from the queue
            current_node = queue.pop(0)
        
        # If the node has no features, mark it as terminal
            if current_node.feature is None:
                current_node.terminal = True
                continue
        
        # Split the tree with the current node
            current_node.split()
        
        # Calculate the degrees of freedom
            degrees_of_freedom = len(current_node.children_values) - 1  
        # Check if chi value calculation is needed
            if current_node.chi != 1 and degrees_of_freedom >= 1:
                chi_value = 0  
            # Count labels in the current node data
                label_counts = {}
                for label in current_node.data[:, -1]:
                    label_counts[label] = label_counts.get(label, 0) + 1
                
            # Calculate probabilities for each label
                total_data_size = current_node.data.shape[0]
                probabilities = {label: count / total_data_size for label, count in label_counts.items()}  
            
            # Iterate over children nodes
                for child_node in current_node.children:
                # Count labels in the child node
                    child_label_counts = {}
                    for label in child_node.data[:, -1]:
                        child_label_counts[label] = child_label_counts.get(label, 0) + 1
                    child_data_size = child_node.data.shape[0]  
                
                # Iterate over label probabilities
                    for label, probability in probabilities.items():
                        expected_count = child_data_size * probability 
                        actual_count = child_label_counts.get(label, 0)
                    # Calculate chi value
                        chi_value += (actual_count - expected_count) ** 2 / expected_count 
            
            # Compare chi value with critical value
                if chi_value < chi_table[degrees_of_freedom][current_node.chi]:
                    current_node.terminal = True
                    continue
        
        # Extend the queue with the children of the current node
            queue.extend(current_node.children)

        return self.root
    


    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        node = self.root ## root of the tree
        end = True  ##the start condition for the loop

        ##if the tree is empty

        if node is None:
            return pred
        ##if the root is a leaf aka there are no children

        if node.terminal:
            return node.pred
        ## coninue as long as we are not a leaf or none
        while not node.terminal and end:
            end = False
            ##get the feature
            feature = instance[node.feature]
            for i, child in enumerate(node.children):
                if node.children_values[i] == feature:
                    end = True
                    node = child
                    break

        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        times_was_equal = 0
        # Traversing over the database
        for instance in dataset:

            # taking the values we need to later compare
            pred_instance = instance[-1]
            pred = self.predict(instance)

                # if our prediction was correct we increase our counter
            if pred == pred_instance:
                times_was_equal = times_was_equal + 1

            # calculating times we were right in total as a %
        accuracy = (times_was_equal / (len(dataset)))
        accuracy = accuracy * 100
        return accuracy
        
    def depth(self):
        return self.root.depth()
    

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """

    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
   # Building the tree according to the max_depth and computing the accuracy
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy,max_depth=max_depth, gain_ratio=True)

        tree.build_tree()
        ##gets the train and test accurecy from the data
        train = tree.calc_accuracy(X_train)
        test = tree.calc_accuracy(X_validation)
        ##appends the train data and test data to their respective arrays
        training.append(train)
        validation.append(test)
    
    return training, validation



def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    chi_val =  [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for chi in  chi_val:
        # Create a new decision tree 
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy,chi = chi, gain_ratio=True)
        tree.build_tree()

        # Calculate training accuracy
        training = tree.calc_accuracy(X_train)  
        chi_training_acc.append(training)

        # Calculate validation accuracy
        validation = tree.calc_accuracy(X_test)  
        chi_validation_acc.append(validation)

        depth.append(depth_of_node(tree.root))
        
    return chi_training_acc, chi_validation_acc, depth



def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    if node is None:
        return 0

    n_nodes = 1  # Count current node
    if node.children:  # Check if node has children
        for child in node.children:
            n_nodes += count_nodes(child)  # Recursively count nodes in children

    return n_nodes







