import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot

class DecisionTreeClassifier:
    def __init__(self, max_depth=10, criterion='gini'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def entropy(self, y):
        unique, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p))

    def gini(self, y):
        unique, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def information_gain(self, X_column, y, threshold):
        if self.criterion == 'gini':
            parent_entropy = self.gini(y)
        else:
            parent_entropy = self.entropy(y)

        left = X_column <= threshold
        right = X_column > threshold

        if len(y[left]) == 0 or len(y[right]) == 0:
            return 0

        n = len(y)
        n_left = len(y[left])
        n_right = len(y[right])

        if self.criterion == 'gini':
            left_e = self.gini(y[left])
            right_e = self.gini(y[right])
        else:
            left_e = self.entropy(y[left])
            right_e = self.entropy(y[right])

        avg_entropy = (n_left / n) * left_e + (n_right / n) * right_e
        return parent_entropy - avg_entropy

    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for f_index in range(X.shape[1]):
            X_column = X[:, f_index]
            thresholds = np.unique(X_column)

            for t in thresholds:
                gain = self.information_gain(X_column, y, t)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f_index
                    best_threshold = t

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if len(y) == 0:
            return self.Node(value=None)

        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            most_common_label = np.bincount(y).argmax()
            return self.Node(value=most_common_label)

        feature, threshold = self.best_split(X, y)

        if feature is None:
            most_common_label = np.bincount(y).argmax()
            return self.Node(value=most_common_label)

        left_i = X[:, feature] <= threshold
        right_i = X[:, feature] > threshold

        left = self.build_tree(X[left_i], y[left_i], depth + 1)
        right = self.build_tree(X[right_i], y[right_i], depth + 1)

        return self.Node(feature, threshold, left, right)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_tree(self, node, x):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_tree(node.left, x)
        else:
            return self.predict_tree(node.right, x)

    def predict(self, X):
        return np.array([self.predict_tree(self.tree, x) for x in X])

    def draw_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree

        indent = "  " * depth
        if node.value is not None:
            print(f"{indent}└── return {node.value}")
            return

        print(f"{indent}├── if x[{node.feature}] <= {node.threshold}:")
        self.draw_tree(node.left, depth + 1)

        print(f"{indent}└── else:")
        self.draw_tree(node.right, depth + 1)
        
    def evaluate(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        print(f"Accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        disp.plot()
        matplotlib.pyplot.show(block=True)

        return accuracy
        