# funkcja do obliczania entropii

#rekurencyjne wybieranie atrybutów, które nie były wcześniej wybierane

#information gain = entropy before - sum(entropy (j,after))

import math
import numpy as np
from sklearn import datasets
from sklearn import model_selection


iris = datasets.load_iris()

def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return sum([-p * math.log(p) for p in y if p > 0])

def information_gain(X_column, y, threshold):
    parent_entropy = entropy(y)
    
    # podział zbioru na dwie części
    left = X_column <= threshold
    right = X_column > threshold        
    
    # jeśli którykolwiek z podziałów jest pusty, to zwracamy 0
    if len(y[left]) == 0 or len(y[right]) == 0:
        return 0
    
    # liczba elementów w lewym i prawym poddrzewie
    n = len(y)
    n_left = len(y[left])
    n_right = len(y[right])
    
    # entropia w lewym i prawym poddrzewie
    left_e = entropy(y[left])
    right_e = entropy(y[right])
    
    # ważona średnia entropii w lewym i prawym poddrzewie
    avg_entropy = (n_left / n) * left_e + (n_right / n) * right_e
    return parent_entropy - avg_entropy

def best_split(X, y):
    best_gain = -1
    best_feature = None
    best_threshold = None
    
    # dla każdej cechy
    for f_index in range(X.shape[1]):
        X_column = X[:, f_index]
        thresholds = np.unique(X_column)
        
        # dla każdego progu
        for t in thresholds:
            gain = information_gain(X_column, y, t)
            if gain > best_gain:
                best_gain = gain
                best_feature = f_index
                best_threshold = t
                
    return best_feature, best_threshold


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, depth=0, max_depth=10):
    if len(y) == 0:
        return Node(value=None)
    
    if len(np.unique(y)) == 1 or depth >= max_depth:
        most_common_label = np.bincount(y).argmax()
        return Node(value=most_common_label)
    
    feature, threshold = best_split(X, y)
    
    if feature is None:
        most_common_label = np.bincount(y).argmax()
        return Node(value=most_common_label)
    
    left_i = X[:, feature] <= threshold
    right_i = X[:, feature] > threshold
    
    left = build_tree(X[left_i], y[left_i], depth + 1, max_depth)
    right = build_tree(X[right_i], y[right_i], depth + 1, max_depth)
    
    return Node(feature, threshold, left, right)
    
def predict_tree(node, x):
    if node.value is not None:
        return node.value
    
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)
    
def predict(tree, X):
    return [predict_tree(tree, x) for x in X]


X = np.array(iris.data)
Y = np.array(iris.target)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

tree = build_tree(X_train, Y_train, max_depth=3)
Y_pred = predict(tree, X_test)

print("Accuracy:", sum(Y_pred == Y_test) / len(Y_test))

# X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [3, 2]])
# y = np.array([0, 0, 0, 1, 1, 1])

# # Budowanie drzewa
# tree = build_tree(X, y, max_depth=3)

# # Przewidywanie
# predictions = predict(tree, X)
# print("Predykcje:", predictions)