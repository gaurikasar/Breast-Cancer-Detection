import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

#read dataset
dataset = pd.read_csv('breast_cancer_bd.csv')
#since there is a ? replacing it
dataset.replace('?', 0, inplace=True)
dataset = dataset.applymap(np.int64)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
#Replacing 2 with 0 and 4 with 1
y_new = []
for i in range(len(y)):
    if y[i] == 2:
        y_new.append(0)
    else:
        y_new.append(1)
y_new = np.array(y_new)
#split training and test data
x_train, x_test, y_train, y_test = train_test_split(X, y_new, test_size=0.25, random_state=0)
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


def weightsbias_initialise(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b


def sigmoid_function(z):
    op = 1/( 1 + np.exp(-z)+0.0000000001 )
    return op

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    y_ = np.dot(w.T,x_train) + b # numeric output of regression algorithm
    y_pre = sigmoid_function(y_) # binary output of sigmoid function
    loss = -y_train*np.log(y_pre)-(1-y_train)*np.log(1-y_pre) # output of loss function
    cost = (np.sum(loss))/x_train.shape[1] # x_train.shape[1]  is for scaling
    # backward propagation
    derivative_w = (np.dot(x_train,((y_pre-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_b = np.sum(y_pre-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_w": derivative_w,"derivative_b": derivative_b}
    return cost,gradients

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    costs1 = []
    costs2 = []
    index = []
    # updating parameters
    for i in range(number_of_iterarion):
        # finding cost and gradient using forward propagation
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        costs1.append(cost)
        # Updating weight & bias
        w = w - learning_rate * gradients["derivative_w"]
        b = b - learning_rate * gradients["derivative_b"]
        if i % 10 == 0:
            costs2.append(cost)
            index.append(i)
    parameters = {"weight": w, "bias": b}

    return parameters, gradients, costs1

#function to print confusion matrix
def confusionmatrix(y_actual,y_prediction):
    return confusion_matrix(y_actual,y_prediction)

#predicting class
def predict(w,b,x_test):
    # forward propogation on x_test array
    z = sigmoid_function(np.dot(w.T,x_test)+b)
    class_predicted= np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            class_predicted[0,i] = 0
        else:
            class_predicted[0,i] = 1

    return class_predicted




def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    dimension = x_train.shape[0]
    w, b = weightsbias_initialise(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)
    model_accur = format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100)
    # Print accuracy

    ypred_ravel = y_prediction_test.ravel()
    ytest_ravel = y_test.ravel()

    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train.T, y_train.T)
    Y_pred = classifier.predict(x_test.T)
    inbuiltaccur = accuracy_score(y_prediction_test.ravel(),Y_pred)
    print("Accuracy of model built from scratch is:  ", model_accur,"%")
    print("Accuracy from inbuilt model using Sklearn library is: ",inbuiltaccur*100,"%")
    print("Confusion Matrix for test data is :")
    print(confusionmatrix(ytest_ravel, ypred_ravel))



logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 100)



