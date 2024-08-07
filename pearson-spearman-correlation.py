

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    
    # Convert x and y to floats
    try:
        x = [float(xi) for xi in x]
        y = [float(yi) for yi in y]
    except ValueError:
        raise ValueError("All elements need to be numeric to compute Pearson correlation")
      
    n = len(x)
      
    # Check if the lengths of x and y are equal
    if n != len(y):
        raise ValueError("The lengths of x and y need to be equal")
      
    # Calculate sums and sums of squares
    sum_of_x = sum(x)
    sum_of_y = sum(y)
    sum_of_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x_squared = sum(xi ** 2 for xi in x)
    sum_y_squared = sum(yi ** 2 for yi in y)
      
    # Calculate numerator and denominator for Pearson correlation coefficient
    numerator = n * sum_of_xy - sum_of_x * sum_of_y
    denominator = ((n * sum_x_squared - sum_of_x ** 2) * (n * sum_y_squared - sum_of_y ** 2)) ** 0.5
      
    # Handle cases where the denominator is zero
    if denominator == 0:
        return 0
      
    # Calculate Pearson correlation coefficient
    r = numerator / denominator
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    # If X is a DataFrame, extract column names or indices
    if isinstance(X, pd.DataFrame):
        name_of_feature = X.columns
        X = X.values  # Convert to numpy array for processing
    else:
        name_of_feature = range(len(X[0]))  # Use indices if not DataFrame

    num_of_features = len(X[0])
    
    # Compute Pearson correlation for each feature with the target
    correlations = []
    for i in range(num_of_features):
        feature_column = [row[i] for row in X]
        try:
            correlation = pearson_correlation(feature_column, y)
            correlations.append((correlation, i))
        except ValueError:
            # Skip non-numeric columns
            continue
    
    # Sort features by absolute value of correlation in descending order
    correlations.sort(key=lambda x: abs(x[0]), reverse=True)
    
    # Select the names or indices of the top n_features
    best_features = [name_of_feature[index] for _, index in correlations[:n_features]]
    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        X = np.insert(X, 0, 1, axis=1)
        # set random seed
        np.random.seed(self.random_state)

        # Initialize theta with random values
        self.theta = np.random.random(X.shape[1])

        for _ in range(self.n_iter):
            # Updating the theta vector
            self.theta = self.theta - self.eta * (np.dot(X.T, ( self.sigmoid( np.dot(X, self.theta)) - y)))

            # Calculate cost and check for convergence.
            self.Js.append(self.cost_function(X, y))
            self.thetas.append(self.theta.copy())
            ##checking if the difference is lower then epsilon
            if len(self.Js) > 1:
              current_cost = self.cost_function(X, y)
              previous_cost = self.Js[-2]
              cost_difference = abs(current_cost - previous_cost)
                
              if cost_difference < self.eps:
                  break

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        X = np.insert(X, 0, 1, axis=1)
        dot_product = np.dot(X, self.theta)
        sigmoid = self.sigmoid(dot_product)
        preds = np.round(sigmoid).astype(int)
        return preds
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    

    def cost_function(self, X, y):
        dot_product = np.dot(X, self.theta)
        sigmoid = self.sigmoid(dot_product)
        epsilon = 1e-5  # small value to avoid log(0).
        term1 = np.dot(y.T, np.log(sigmoid + epsilon))
        term2 = np.dot((1 - y).T, np.log(1 - sigmoid + epsilon))
        J = (-1.0 / len(y)) * (term1 + term2)
        return J

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

     # Set random seed
    np.random.seed(random_state)

    # Shuffle data indices
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    X = X[indexes]
    y = y[indexes]

    # Calculate fold size
    fold_size = X.shape[0] // folds
    accuracies = []

    for i in range(folds):
        # Split the data
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        y_test = y[test_start:test_end]
        X_test = X[test_start:test_end]

        train_indices_left = np.arange(test_start)
        train_indices_right = np.arange(test_end, X.shape[0])
        train_indices = np.concatenate((train_indices_left, train_indices_right))
        y_train = y[train_indices]
        X_train = X[train_indices]

        # Fit the algorithm on the training data
        algo.fit(X_train, y_train)

        # Predict the labels
        y_prediction = algo.predict(X_test)

        # Calculate accuracy
        accuracy = np.mean(y_prediction == y_test)
        accuracies.append(accuracy)

    # Compute the mean cross-validation accuracy
    cv_accuracy = np.mean(accuracies)
    
    return cv_accuracy


def calculate_accuracy(y_pred, y_test):
    precision = np.mean(y_pred == y_test)
    return precision




def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    # Calculate the exponent of the normal distribution formula
    exponent = np.square((data - mu)) / (-2 * np.square(sigma))
    
    # Calculate the normal distribution without the exponent
    pdf = np.exp(exponent)
    
    # Normalize the result by dividing by the standard deviation and the square root of 2*pi
    pdf = pdf / (np.sqrt(2 * np.pi * np.square(sigma)))
    
    return pdf
class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        indices = np.random.choice(data.shape[0], self.k, replace=False)
        self.sigmas = np.random.random_integers(self.k)
        self.weights = np.ones(self.k) / self.k
        self.mus = data[indices].reshape(self.k)

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        # calculate responsability according to formula
        norm = norm_pdf(data, self.mus, self.sigmas)
        responsabilities = self.weights * norm
        sum_responsability = np.sum(responsabilities,axis=1,keepdims=True)
        self.responsibilities=responsabilities/sum_responsability

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        # maximizes according to the formula
        self.weights = np.mean(self.responsibilities, axis=0)
        self.mus = np.sum(self.responsibilities * data.reshape(-1,1), axis=0)
        self.mus = self.mus / np.sum(self.responsibilities, axis=0)

        # Calculate the weighted sum of squared differences for each component
        weighted_squared_diff = self.responsibilities * np.square(data.reshape(-1, 1) - self.mus)

        # Calculate the variance for each component
        variance = np.mean(weighted_squared_diff, axis=0)
        
        self.sigmas = np.sqrt(variance  / self.weights)

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        self.costs.append(self.cost(data))
        
        # Finding the parameters for the distribution.
        # The process stops when the difference between the previous and current cost is less than epsilon,
        # or when we reach n_iter.  
        for _ in range(self.n_iter): 
            cost = self.cost(data)
            self.costs.append(cost)
            self.expectation(data)  
            self.maximization(data)  
            if self.costs[-1] > cost and self.costs[-1] - cost < self.eps:
                self.costs.append(cost)
                break
            self.costs.append(cost)

    # Calculating the cost of the data.
    def cost(self, data):
        total_cost = 0
        costs = self.weights * norm_pdf(data, self.mus, self.sigmas)
        
        for i in range(len(data)):
            total_cost += costs[i]

        log_total_cost = np.log(total_cost)
        negative_log_total_cost = -np.sum(log_total_cost)
        
        return negative_log_total_cost

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
     # Reshape data to handle vectorized operations
    data_reshaped = data.reshape(-1, 1)
    
    # Calculate the GMM pdf
    pdf = np.sum(weights * norm_pdf(data_reshaped, mus, sigmas), axis=1)
    
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.gausian = None

    def fit(self, X, y):
          """
          Fit training data.

          Parameters
          ----------
          X : array-like, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features.
          y : array-like, shape = [n_examples]
            Target values.
          """
          # Store the training data and labels
          self.X = X
          self.y = y
          self.num_Of_Instances = len(X)
          
          # Calculate priors for each class
          self.priors = {}
          for class_Label in np.unique(y):
              self.priors[class_Label] = len(y[y == class_Label]) / len(y)

          # Initialize Gaussian distributions for each class and feature
          self.gaussians = {}
          for class_Label in np.unique(y):
              self.gaussians[class_Label] = {}
              for feature in range(X.shape[1]):
                  # Initialize an EM (Expectation-Maximization) model for each feature
                  self.gaussians[class_Label][feature] = EM(self.k)

          # Fit Gaussian distributions for each class and feature
          for label in self.gaussians.keys():
              for feature in self.gaussians[label].keys():
                  # Extract the feature data for the current class
                  X_label_feature = X[y == label][:, feature].reshape(-1, 1)
                  # Fit the Gaussian distribution using EM
                  self.gaussians[label][feature].fit(X_label_feature)


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []  # Initialize an empty list for storing predictions

        # Iterate through each instance in X
        for instance in X:
            posterior_values = []

            # Calculate posterior probability for each class label
            for class_label in self.priors.keys():
                likelihood = 1

                # Calculate likelihood for the current instance and class label
                for feature_idx in range(X.shape[1]):
                    weights, mus, sigmas = self.gaussians[class_label][feature_idx].get_dist_params()
                    gmm = gmm_pdf(instance[feature_idx], weights, mus, sigmas)
                    likelihood *= gmm
                
                # Calculate posterior probability by multiplying likelihood with prior
                posterior = likelihood * self.priors[class_label]
                posterior_values.append((posterior, class_label))
            
            # Choose the class label with the maximum posterior probability
            max_posterior_class = max(posterior_values, key=lambda t: t[0])[1]
            preds.append(max_posterior_class)  # Append the predicted class label to preds

        return np.array(preds).reshape(-1,1)  # Return the list of predicted class labels
    

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 
  
    # Initialize variables to store accuracies
    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    def calc_accuracy(prediction , test):
   
      return np.mean(prediction == test)
    
    # Fit Logistic Regression model
    log_regression = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    log_regression.fit(x_train, y_train)
    # accuracies for Logistic Regression
    lor_train_acc = calc_accuracy(y_train, log_regression.predict(x_train))
    lor_test_acc = calc_accuracy(y_test, log_regression.predict(x_test))
    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=log_regression, title="Decision Regressor using Logistic Regression")

    # Fit Naive Bayes model

    naive_b = NaiveBayesGaussian()
    naive_b.fit(x_train, y_train)
    # accuracies for Naive Bayes
    bayes_train_acc = calc_accuracy(y_train.reshape(-1, 1), naive_b.predict(x_train))
    bayes_test_acc = calc_accuracy(y_test.reshape(-1, 1), naive_b.predict(x_test))
    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=naive_b, title="Decision Regressor using Naive Bayes")
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(log_regression.Js)), log_regression.Js)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iteration for Logistic Regression')
    plt.grid(True)
    plt.show()
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}



# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ['blue', 'red']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    plt.show()

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None


    def generate_data(num_instances, means, covariance, labels):
        dataset_features = np.empty((num_instances, 3))
        dataset_labels = np.empty((num_instances))
        gaussian_size = num_instances // len(means)

        for i, mean in enumerate(means):
            label = labels[i]
            points = np.random.multivariate_normal(mean, covariance, gaussian_size)
            dataset_features[i * gaussian_size: (i + 1) * gaussian_size] = points
            dataset_labels[i * gaussian_size: (i + 1) * gaussian_size] = np.full(gaussian_size, label)
        return dataset_features, dataset_labels

    # Dataset A parameters
    dataseta_covariance = [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]
    dataseta_labels = [1, 0, 0, 1]
    dataseta_means = [[0, 0, 0], [4, 4, 4], [12, 12, 12], [18, 18, 18]]

    # Generate dataset A
    dataset_a_features, dataset_a_labels = generate_data(5000, dataseta_means, dataseta_covariance, dataseta_labels)

    # Dataset B parameters
    datasetb_covariance = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
    datasetb_labels = [0, 1]
    datasetb_means = [[0, 5, 0], [0, 7, 0]]

    # Generate dataset B
    dataset_b_features, dataset_b_labels = generate_data(5000, datasetb_means, datasetb_covariance, datasetb_labels)

    return {
        'dataset_a_features': dataset_a_features,
        'dataset_a_labels': dataset_a_labels,
        'dataset_b_features': dataset_b_features,
        'dataset_b_labels': dataset_b_labels
    }