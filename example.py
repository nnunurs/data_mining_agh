from decision_tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_wine
from sklearn import model_selection
import numpy as np

# Load the datasets
iris = load_iris()
wine = load_wine()

# Iris
print("Iris:\n")
X, y = np.array(iris.data), np.array(iris.target)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(max_depth=20, criterion='gini')
clf.fit(X_train, y_train)

print("Tree:")
clf.draw_tree()
clf.evaluate(X_train, y_train, X_test, y_test)

# Wine
print("\nWine:")
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

print("Tree:")
clf.draw_tree()
clf.evaluate(X_train, y_train, X_test, y_test)

#TODO mist
# https://paperswithcode.com/dataset/mist