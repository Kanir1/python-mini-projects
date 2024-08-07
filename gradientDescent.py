

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
     # Calculate the range of each feature in X
    X_diff = np.max(X, axis=0) - np.min(X, axis=0)
    
    # Calculate the range of y
    y_diff = np.max(y, axis=0) - np.min(y, axis=0)

    # Calculate the mean of each feature in X
    X_mean = np.mean(X, axis=0)
    
    # Calculate the mean of y
    y_mean = np.mean(y, axis=0)

    # Mean normalization for features
    X = (X - X_mean) / X_diff
    
    # Mean normalization for labels
    y = (y - y_mean) / y_diff
    
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    # Creating a unity vector with the size of X
    new_len= len(X)
    column_of_ones = np.ones((new_len))

    # Stack the column_of_ones to the left of X to create X with an additional column of ones
    X = np.column_stack((column_of_ones, X))

    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
     # Number of training examples
    m = len(X)
    J = 0
    
    # Compute prediction
    prediction = np.dot(X, theta)
    
    # Compute squared differences
    squared_diff = (prediction - y) ** 2
    
    # Compute cost
    J = np.sum((squared_diff) / (2 * m))
    
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    # Make a copy of theta to avoid changing the original outside the function
    theta = theta.copy()
    
    # Initialize a list to store the cost value in every iteration
    J_history = []

    m = len(X)  # Number of training examples

    # Gradient descent iterations
    for i in range(num_iters):
        # Calculate prediction
        prediction = np.dot(X, theta)
        
        # Compute error
        error = prediction - y
        
        # Update theta using gradient descent formula
        theta = theta - (alpha / m) * np.dot(X.T, error)
        
        # Compute and store the cost value
        add_cost = compute_cost(X, y, theta)
        J_history.append(add_cost)

    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    
   # Transpose the input matrix X
    X_transpose = X.T
    
    # Compute X_transpose_mult_X (X transpose multiplied by X)
    X_transpose_mult_X = np.dot(X_transpose, X)
    
    # Compute the inverse of X_transpose_mult_X using np.linalg.inv
    X_transpose_mult_X_inverse = np.linalg.inv(X_transpose_mult_X)
    
    # Compute the optimal parameters using the pseudo-inverse approach
    pinv_theta = np.dot(X_transpose_mult_X_inverse, X_transpose).dot(y)

    return pinv_theta
def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    theta = theta.copy()

    # Initialize an empty list to store the cost values for each iteration
    
    J_history = [] 
    # Number of training examples
    m = len(X)
    # Iterate through the specified number of iterations
    for i in range(num_iters):
         # Compute the predictions using the current parameter values
        prediction = X.dot(theta)
         # Calculate the error between the predictions and the actual labels
        error = prediction - y
        # Update the parameters (theta) using gradient descent
        theta = theta - (alpha / m) * np.dot(X.T, error)
         # Compute the cost for the updated parameters
        add_cost = compute_cost(X, y, theta)
         # Append the cost to the history list
        J_history.append(add_cost)
         
    # Check if the change in cost becomes negligible
        if i > 0 and J_history[i - 1] - J_history[i] < 1e-8:
            break
    # Return the optimized parameters (theta) and the history of cost values
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}

    # we want to iterate over each alpha and check which one is best for us
    # as we insert the alpha alongside our theta vector into the efficient gradient descent
    for alpha in alphas:
        # we will pick a random vector to be our theta vector
        thetaVector = np.random.random(size= X_train.shape[1])
        #we take [0] because it returns a tuple, and we need the first element which is our theta vector
        thetaVector = efficient_gradient_descent(X_train,y_train,thetaVector,alpha,iterations)[0]
        # after we found our theta vector we will now send it to the cost function along x and y to get the validation loss and put it in the dictionary
        valuetoinsert = compute_cost(X_val, y_val, thetaVector)
        alpha_dict[alpha] = valuetoinsert

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    xValueofFeatures = 0
    xTrainofFeatures = 0
    for i in range(5):
        #defining arbitary theta vector and our cost function array from which we will select our 5 minimum values
        thetaVector = np.random.random(size=i + 2)
        costFunctions = []
        # now we will iterate over every feature in x, apply bias trick and recieve theta from the gradient descent
        for k in range(X_train.shape[1]):           
            if k not in selected_features:
                #we will add our current feature to the list temporarly
                selected_features.append(k)
                # training x based on the selected features as required
                xValueofFeatures = apply_bias_trick(X_val[:, selected_features])
                xTrainofFeatures = apply_bias_trick(X_train[:, selected_features])
                #aquiring theta vector from the efficient gradient descent, we use [0] because its a tuple
                actualtheta = efficient_gradient_descent(xTrainofFeatures, y_train, thetaVector, best_alpha, iterations)[0]
                #computing the cost
                costOfFunction = compute_cost(xValueofFeatures,y_val,actualtheta)
                #appending the cost of the function based on the selected feature, alongside the feature
                costFunctions.append((costOfFunction,k))
                #removing k after adding it temporarly
                selected_features.pop()


        #we retrieve the feature of our costFunctions dictionary based on the minimal value of cost(x[0]), and we take the feature alone ([1])      
        #we then append it to our selected features, and repeat it 5 times
        selected_features.append(min(costFunctions, key = lambda x:x[0])[1])

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    newlistofSquaredFeature = []
    squaredString = ""
     #we iterate over every collumn to multiply each feature with itself and all the other features
    for column in df_poly.columns:
        #index of the current column
        i = df_poly.columns.get_loc(column)
        #second loop
        for otherColumn in df_poly.columns[i:]:
            #the squared column that we will insert
            columnToinsert = df_poly[column] * df_poly[otherColumn]
            if column == otherColumn:
                squaredString = column + '^2'
            else:
                squaredString = column + '*' + otherColumn
            columnToinsert.name = squaredString
            newlistofSquaredFeature.append(columnToinsert)
    #we now concat to our dataframe the squared features with their string alongside the existing features
    df_poly = pd.concat([df_poly] + newlistofSquaredFeature, axis=1)

    return df_poly