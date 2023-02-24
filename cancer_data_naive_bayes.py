
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.naive_bayes import GaussianNB


def read_data(file):
    col_names = ['id_number','clump_thickness','cell_size_uniformity','cell_shape_uniformity','marginal_adhesion','epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
    data = pd.read_csv(file, header = None, names=col_names)
    # replacing the value with an integer, we can use any value here, but since these are outliers, need to indicate the same with the data 
    data.replace('?', -99999, inplace = True)
    data = data.applymap(np.int64)
    #Drop the id_number column as it is not required to train and test the model
    input_data = data.drop(['id_number'], axis = 1, inplace = False)

    X = input_data.loc[:,input_data.columns!='class']
    y=input_data.loc[:,"class"].map({2:0, 4:1})

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25, random_state = 122)
    return X_train,X_test,y_train,y_test

if __name__ == "__main__":
    # read the file
    X_train,X_test,y_train,y_test = read_data("/Users/dhatri/workplace/ML/project/breast-cancer-wisconsin.data.txt")

    # Fit a Naive Bayes Classifier from sklearn
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    ### Train data Metrics
    train_data_preds = classifier.predict(X_train)

    train_cf = confusion_matrix(y_train, train_data_preds)
    print("Train data Confusion Matrix : \n", train_cf)

    sns.heatmap(train_cf, annot=True, cmap='Blues', fmt='g')
    print("\nTrain data Accuracy",accuracy_score(y_train,train_data_preds))

    print(classification_report(y_train, train_data_preds, labels=[2,4]))

    ### Test data Metrics
    test_data_preds = classifier.predict(X_test)

    test_cf = confusion_matrix(y_test, test_data_preds)
    print("Test data Confusion Matrix : \n", test_cf)
    sns.heatmap(test_cf, annot=True, cmap='Blues', fmt='g')


    print("\nTest data Accuracy",accuracy_score(y_test,test_data_preds))
    print(classification_report(y_test, test_data_preds, labels=[0,1]))
