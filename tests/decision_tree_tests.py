import pytest
from decision_tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def test_decision_tree():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=10, criterion='gini')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc > 0.9

def test_decision_tree_sklearn():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=10, criterion='gini')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc > 0.9

def test_entropy():
    model = DecisionTreeClassifier(max_depth=10, criterion='entropy')
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    ent = model.entropy(y)
    assert ent == 1.0

def test_gini():
    model = DecisionTreeClassifier(max_depth=10, criterion='gini')
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    g = model.gini(y)
    assert g == 0.5

def test_information_gain():
    model = DecisionTreeClassifier(max_depth=10, criterion='gini')
    X_column = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    ig = model.information_gain(X_column, y, 5)
    assert ig == 0.5

def test_best_split():
    model = DecisionTreeClassifier(max_depth=10, criterion='gini')
    X = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 1], [7, 1], [8, 1], [9, 1], [10, 1]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    f, t = model.best_split(X, y)
    assert f == 0
    assert t == 5

if __name__ == '__main__':
    pytest.main()