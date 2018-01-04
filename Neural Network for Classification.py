"""
@author: Shubham Shantaram Pawar
"""

# importing all the required libraries
import numpy as np
import matplotlib .pyplot as plt
from sklearn.datasets import load_iris

# function to plot the training data
def plotTrainingData(X, y):
    versicolor = np.where(y==0)
    verginica = np.where(y==1)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Scatter Plot of Training Data')
    ax.scatter(X[versicolor][:,0], X[versicolor][:,1], color='blue', label='versicolor', marker='o')
    ax.scatter(X[verginica][:,0], X[verginica][:,1], color='red', label='verginica', marker='+')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('petal length (cm)')
    ax.set_ylabel('petal width (cm)')
    ax.legend()
    fig.set_size_inches(10, 6)
    fig.show()

# function to plot cost vs iterations
def plotCostVsIterations(J_history, iterations):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('cost vs iterations')
    ax.set_xlabel(r'iterations')
    ax.set_ylabel(r'$J(\theta)$')
    ax.scatter(range(iterations), J_history, color='blue', s=10)
    fig.set_size_inches(8, 5)
    fig.show()
    
# function to initialize parameters to be uniformly distributed random numbers
# between -0.22 and 0.22
def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.22
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W

# function to calculate sigmoid of activity
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# function to calculate sigmoid gradient
def sigmoidGradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

# function to compute cost and gradients
def computeCost(X, y, Theta1, Theta2):
    m, n = X.shape
    
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    # Forward Propagation:
    
    # input layer values (with bias unit)
    a1 = X
    # calculating activity of hidden layer
    z2 = np.dot(a1, Theta1.T)
    a, b = z2.shape
    # calculating activation of hidden layer (with bias unit)
    a2 = np.concatenate((np.ones((a, 1)), sigmoid(z2)), axis=1)
    # calculating activity of output layer
    z3 = np.dot(a2, Theta2.T)
    # calculating activation of output layer
    a3 = sigmoid(z3)
    # hypothesis
    h = a3
    
    # calculating mean squared error cost
    J = (1 / m) * np.sum(np.sum(-1 * np.multiply(y, np.log10(h)) - np.multiply(1 - y, np.log10(1 - h)), axis=0))

    # Backpropagation:
    
    # calculating gradients
    d3 = h - y
    d2 = np.multiply(d3 * Theta2,  sigmoidGradient(np.concatenate((np.ones((a, 1)), z2), axis=1)))
    c, d = d2.shape
    d2 = d2[:, [1, d-1]]
    
    delta1 = d2.T * a1
    delta2 = d3.T * a2
    
    Theta1_grad = delta1 / m
    Theta2_grad = delta2 / m
    
    return J, Theta1_grad, Theta2_grad

# function for gradient descent
def gradientDescent(x, y, Theta1, Theta2, alpha, num_iters):
    
    # initializing matrix to store cost history
    J_history = np.zeros((num_iters,1))
    
    for iter in range(0, num_iters):
        
        J, Theta1_grad, Theta2_grad = computeCost(x, y, Theta1, Theta2)
    
        #updating parameters/thetas
        Theta1 = np.subtract(Theta1, alpha * Theta1_grad)
        Theta2 = np.subtract(Theta2, alpha * Theta2_grad)
        
        J_history[iter] = J
        
    return J_history, Theta1, Theta2

# function to make a 100 folds of the data for leave-one-out analysis 
def leaveOneOut_split(X, y):
    k_folds = 100
    data_splits = []
    
    for i in range(k_folds):
        temp = []
        train_data = {}
        index = list(range(k_folds))
        index.pop(i)
        train_data['X'] = X[index]
        train_data['y'] = y[index]
        test_data = {}
        test_data['X'] = X[i]
        test_data['y'] = y[i]
        temp.append(train_data)
        temp.append(test_data)
        data_splits.append(temp)
    
    return data_splits

# function to perform leave-one-out analysis 
def leaveOneOutAnalysis(X, y, alpha, iterations, input_layer_size, hidden_layer_size, output_layer_size):
    total_error = 0
    data_splits = leaveOneOut_split(X, y)
    
    for i, data_split in enumerate(data_splits):
        
        print('\nTraining with fold ' + str(i+1) + '...')
        
        X_train = data_split[0]['X']
        y_train = data_split[0]['y']
        
        X_test = data_split[1]['X']
        y_test = data_split[1]['y'][0, 0]
        
        # initializing parameters/thetas
        theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        theta2 = randInitializeWeights(hidden_layer_size, output_layer_size)
        
        J_history, Theta1, Theta2 = gradientDescent(X_train, y_train, theta1, theta2, alpha, iterations)
        
        # forward propagation for prediction
        a1 = X_test
        z2 = np.dot(a1, Theta1.T)
        a, b = z2.shape
        a2 = np.concatenate((np.ones((a, 1)), sigmoid(z2)), axis=1)
        z3 = np.dot(a2, Theta2.T)
        a3 = sigmoid(z3)
        h = a3
        
        # predicting class label for the test data
        if h >= 0.5:
            y_predict = 1.0
        else:
            y_predict = 0.0
        
        # comparing predicted class label with the test/actual class label
        # if not equal, increase total error by 1
        if y_predict != y_test:
            total_error += 1
            
    return total_error/100

def main():
    
    input_layer_size = 2
    hidden_layer_size = 2
    output_layer_size = 1
    
    # loading iris dataset
    iris = load_iris()
    
    # selecting indices for samples corresponding to versicolor and virginica classes respectively
    versicolor_target = np.where(iris.target==1)
    virginica_target = np.where(iris.target==2)
    
    # extracting training dataset corresponding to versicolor and virginica classes
    X_train = iris.data[np.concatenate((versicolor_target[0], virginica_target[0]), axis = 0)][:, [2, 3]]
    y_train = iris.target[0:100]
    
    # ploting training data
    plotTrainingData(X_train, y_train)
    
    # min-max normalization/scaling
    X_train[:, 0] = (X_train[:, 0] - np.min(X_train[:, 0])) / (np.max(X_train[:, 0]) - np.min(X_train[:, 0]))
    X_train[:, 1] = (X_train[:, 1] - np.min(X_train[:, 1])) / (np.max(X_train[:, 1]) - np.min(X_train[:, 1]))
    
    m, n = X_train.shape
    
    # adding one's for the bias term
    X = np.concatenate((np.ones((m, 1)), X_train), axis=1)
    
    y = np.matrix(y_train).reshape(100,1)
        
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size)
    
    # no. of iterations
    iterations = 5000
    
    # learning rate
    alpha = 0.1
    
    print('\nPerforming logistic regression using an ANN on the entire dataset...')
      
    J_history, Theta1, Theta2 = gradientDescent(X, y, initial_Theta1, initial_Theta2, alpha, iterations)
    
    # plotting cost vs iterations
    plotCostVsIterations(J_history, iterations)
    
    print('\nTheta 1:')
    print(Theta1)
    print('\nTheta 2:')
    print(Theta2)
    
    # computing average error rate for the model using leave-one-out analysis
    avg_error = leaveOneOutAnalysis(X, y, alpha, iterations, input_layer_size, hidden_layer_size, output_layer_size)
    
    print('\nThe average error rate for the ANN model after performing leave-one-out analysis is ' + str(avg_error) +'.')
        
if __name__ == '__main__':
    main()