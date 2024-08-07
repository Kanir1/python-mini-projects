

import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.3,
            (1, 1): 0.4
        }

        self.X_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }

        self.X_Y_C = {
            (0, 0, 0): 0.045,
            (0, 0, 1): 0.045,
            (0, 1, 0): 0.105,
            (0, 1, 1): 0.105,
            (1, 0, 0): 0.105,
            (1, 0, 1): 0.105,
            (1, 1, 0): 0.245,
            (1, 1, 1): 0.245
        }

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ## from the definition if dependency, p(x)*p(y) = p(x,y) iff they are independent
        for x, y in X_Y.keys():
            if X[x] * Y[x] == X_Y[(x,y)]:
                return False
        return True
        

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ##Go over the database
        for x in X:
            for y in Y:
                for c in C:
                    if not np.isclose(
                        X_Y_C[(x, y, c)],
                        ( X_C[(x, c)] * Y_C[(y, c)]) /C[c],
                    ):
                        return False

        return True
        

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    e_Powered_by_Lamda = np.exp(-rate) ## to get e^-lamda
    lamda_Powered_by_K = rate ** k ## to get lamda^k
    k_Factorial = np.math.factorial(k) ## to get k!

    formula = ((lamda_Powered_by_K*e_Powered_by_Lamda)/k_Factorial) ## the formula of poisson 
    log_p = np.log(formula) 
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.zeros(len(rates)) ## initializes the 1d array
     ## iterates over the rates object to get each lamda
    likelihoods = [sum(poisson_log_pmf(sample, lamda) for sample in samples) for lamda in rates]
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    ## calculating according to formula
    index = np.argmax(likelihoods)
    rate = rates[index]
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ## compute the mean of the samples data
    mean = np.mean(samples)
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ## calculating variables according to the formula
    std_powered_by_2 = 2 * ((std) ** 2)
    x_Minus_mean = -((x - mean) ** 2)
    ## calculating the exp
    exp = np.exp(x_Minus_mean/std_powered_by_2)
    ## calculating the denominator
    denominator = std * np.sqrt(2 * np.pi)
    ## calculating the normal pdf
    p = exp/denominator
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.database = None
        self.std = None
        self.class_database = None
        self.mean = None

        # The fields of the object.
        self.database = np.copy(dataset)
        class_database = dataset[dataset[:, -1] == class_value]
        class_database = class_database[:, :-1]
        self.class_database = class_database
        self.std = np.std(class_database, axis=0)
        self.mean = np.mean(class_database, axis=0)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        total_amount_samples = len(self.database) ## get the total amount of data
        class_amount_samples = len(self.class_database) ## get the amount of the class spesific data

        prior = class_amount_samples/total_amount_samples ## calculate the probabilty
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        ## we want to use the normal pdf function we wrote earlier, for each collumn except the class collum
        data_range = len(self.database.T) -1

        for i in range(data_range):
            likelihood = likelihood * normal_pdf(x[i], self.mean[i], self.std[i])  

        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        prior = self.get_prior()
        likelihood = self.get_instance_likelihood(x)
        # Calculating the posterior probability
        posterior = likelihood * prior
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        #declaring the fields
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        post1 = self.ccd1.get_instance_posterior(x)
        post0 = self.ccd0.get_instance_posterior(x)

        # Check if the posterior probability of class 0 is higher
        if post0 <= post1:
            pred = 1
        else:
            pred = 0
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    test_set_length = len(test_set)
    counter = 0
    for instance in test_set:
        ## gets the actual class of the instance
        actual_class = instance[-1]
        all_features = instance[:-1]
        class_prediction = map_classifier.predict(all_features)
        ## if we were right, increment the counter by 1 
        if actual_class == class_prediction:
            counter = counter + 1
    ## checks how many times we were right
    acc = counter / test_set_length

    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    n = x.shape[0]
    ## declaring variables needed for the formula, e.g inverse matrix, difference, and determinant of the cov matrix
    inverse_matrix  = np.linalg.inv(cov)
    determinant = np.linalg.det(cov)
    difference = (x - mean).reshape((n, 1))
    exponent = -0.5 * difference.T @ inverse_matrix @ difference

    # calculating according to the Multivariate Normal Distributions
    exponential_term = np.exp(exponent)
    normalization_factor = 1.0 / np.sqrt((2 * np.pi) ** n * determinant)
    pdf = normalization_factor * exponential_term
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.class_value = class_value
        self.dataset = dataset

        class_indices = dataset[:, -1] == class_value
        class_database = dataset[class_indices, :-1]

        self.mean = np.mean(class_database, axis=0)
        database_transpose = class_database.T
        self.covariance = np.cov(database_transpose)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        sample_length = np.shape(self.mean)[0]
        corrected_samples = 0

        for instance in self.dataset:
          ## checks if we had a correct prediction
          if instance[-1] == self.class_value:
             corrected_samples = corrected_samples + 1

        # calculate how many times we were right out of the whole database
        prior = corrected_samples / sample_length
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ## sending the variables to the multi normal function
        likelihood = multi_normal_pdf(x, self.mean, self.covariance)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        prior = self.get_prior()
        likelihood = self.get_instance_likelihood(x)

        # Calculating the posterior probability
        posterior = likelihood * prior
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd1 = ccd1
        self.ccd0 = ccd0
        
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        prior1 = self.ccd1.get_prior()
        prior0 = self.ccd0.get_prior()

       # Check if the posterior probability of class 0 is lower or equal
        if prior0 <= prior1:
            pred = 1
        else:
            pred = 0
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd1 = ccd1
        self.ccd0 = ccd0

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        likelihood1 = np.prod(self.ccd1.get_instance_likelihood(x))
        likelihood0 = np.prod(self.ccd0.get_instance_likelihood(x))

       # Check if the likelihood probability of class 0 is lower or equal
        if likelihood0 <= likelihood1:
            pred = 1
        else:
            pred = 0

        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.database = np.copy(dataset)
        self.class_value = class_value
        condition = dataset[:, -1] == class_value
        self.class_data = dataset[condition]
        self.feature_amount = len(self.database.T) - 1
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        instance_class = len(self.class_data)
        database = len(self.database)

        # Calculating the prior probability
        prior = instance_class / database
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = 1.0
        num_instances_class = len(self.class_data)

    # Iterating over the data to gather necessary information.
        for i, feature_column in enumerate(self.class_data.T[:-1]):
            matching_indices = x[i] == feature_column[:]
            # Filtering the feature column based on the matching indices.
            matching_values = feature_column[matching_indices]
            # Counting the number of matching instances.
            num_matching_instances = len(matching_values)
            # Counting the number of unique values in the feature column.
            unique_values = set(feature_column)
            num_unique_values_feature = len(unique_values)

            # Calculation of the likelihood
            if num_instances_class + num_unique_values_feature == 0:
                likelihood *= (num_matching_instances + 1) / (num_instances_class + num_unique_values_feature + EPSILLON)
            else:
                likelihood *= (num_matching_instances + 1) / (num_instances_class + num_unique_values_feature)
                
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        prior = self.get_prior()
        likelihood = self.get_instance_likelihood(x)

        # Calculating the posterior probability
        posterior = likelihood * prior
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd1 = ccd1
        self.ccd0 = ccd0

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        post1 = self.ccd1.get_instance_posterior(x)
        post0 = self.ccd0.get_instance_posterior(x)

        # Check if the posterior probability of class 0 is higher
        if post0 <= post1:
            pred = 1
        else:
            pred = 0
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        test_set_length = len(test_set)
        counter = 0
        for instance in test_set:
            ## gets the actual class of the instance
            actual_class = instance[-1]
            all_features = instance[:-1]
            class_prediction = self.predict(all_features)
            ## if we were right, increment the counter by 1 
            if actual_class == class_prediction:
                counter = counter + 1
        ## checks how many times we were right
            acc = counter / test_set_length
        return acc


