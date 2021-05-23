#created and edited by Samuel Phillips

#imports for data, classes and more
from pandas import DataFrame
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
from matplotlib import pyplot as plt

#-- a2p1 starts here --
#iris data is loaded
iris = load_iris()
iData = iris.data

#data is collected from the iris dataset
X = iris.data[:, :2]
y = iris.target

print()

#train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
n = KNeighborsClassifier()
n.fit(X_train, y_train)
p1 = n.predict(X_test)

#0 to 49 -> Setosa
#50 to 99 -> Versicolor
#100 to 149 -> Virginica

#scatterplot size is set
plt.figure(figsize=(8,8))

#first points of each type of flower are plotted so the legend can show correctly
plt.scatter(X[:, :1][0], X[:, 1:][0], facecolors='none', edgecolors='red', label='setosa')
plt.scatter(X[:, :1][50], X[:, 1:][50], facecolors='none', edgecolors='green', label='versicolor')
plt.scatter(X[:, :1][100], X[:, 1:][100], facecolors='none', edgecolors='blue', label='virginica')

#for loop that plots all the points for sepal length and width
for i in range(0, len(X)):
    if i < 50:
        plt.scatter(X[:, :1][i], X[:, 1:][i], facecolors='none', edgecolors='red')
    elif i < 100 and i > 49:
        plt.scatter(X[:, :1][i], X[:, 1:][i], facecolors='none', edgecolors='green')
    else:
        plt.scatter(X[:, :1][i], X[:, 1:][i], facecolors='none', edgecolors='blue')

#lists to hold x and y values for correct and incorrect predictions
corrX, corrY = [], []
incorX, incorY = [], []

#for loop that collects the x and y values of the correct and incorrect predictions
for i in range(0, len(p1)):
    if p1[i] == y_test[i]:
        corrX.append(X_test[:, :1][i])
        corrY.append(X_test[:, 1:][i])

    elif p1[i] != y_test[i]:
        incorX.append(X_test[:, :1][i])
        incorY.append(X_test[:, 1:][i])       
        
#first point of the correct prediciton x and y coordinates is plotted so the legend can show correctly
plt.scatter(corrX[0], corrY[0], color='black', marker=(5, 1), label='correct prediction')
plt.scatter(incorX[0], incorY[0], color='hotpink', marker=(5, 1), label='incorrect prediction')

#collection of all the correct points
for i in range(0, len(corrX)):
    plt.scatter(corrX[i], corrY[i], color='black', marker=(5, 1))
    plt.scatter(incorX[0], incorY[0], color='hotpink', marker=(5, 1))

#scatterplot legend is made along with x and y axis names
plt.legend()
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width') 

#plot is showed and saved to pdf
plt.show()
plt.savefig('a2p1_scatter.pdf')
#-- a2p1 ends here --

#-- a2p2 starts here --
#iris data is loaded and set
iris = load_iris()
X, y = load_iris(return_X_y=True)

#train-test-split is created from the iris data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
n = tree.DecisionTreeClassifier()
n = n.fit(X_train, y_train)

#tree is created
plt.figure(figsize=(8,8))
tree.plot_tree(n, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
plt.savefig('a2p2_dtree.pdf')

#list for prediction accuracy is made
prediction_accuracy = []

#for loop that makes a train-test-split ten times and stores each prediction accuracy into the pa list
for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    n = tree.DecisionTreeClassifier()
    n.fit(X_train, y_train)
    yp = n.predict(X_test)
    prediction_accuracy.append(metrics.accuracy_score(y_test, yp))

#prediction accuracy list is printed
print(prediction_accuracy)

#mean of all the prediction accuracy values is printed out 
print(np.mean(prediction_accuracy))
#-- a2p2 ends here --