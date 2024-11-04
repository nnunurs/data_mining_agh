import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class DecisionTreeClassifier:
    def __init__(self, max_depth=10, criterion='gini', min_samples_split=2, verbose=False):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.verbose = verbose
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

        if self.verbose:
            print(f"Best split: Feature {best_feature} at threshold {best_threshold} with gain {best_gain}")
        
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if self.verbose:
            print(f"Building tree at depth {depth} with {len(y)} samples")

        if len(y) < self.min_samples_split:
            most_common_label = Counter(y).most_common(1)[0][0]
            if self.verbose:
                print(f"Not enough samples to split. Creating leaf node with class {most_common_label}")
            return self.Node(value=most_common_label)

        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            most_common_label = Counter(y).most_common(1)[0][0]
            if self.verbose:
                print(f"Leaf node at depth {depth} with class {most_common_label}")
            return self.Node(value=most_common_label)

        feature, threshold = self.best_split(X, y)

        if feature is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            if self.verbose:
                print(f"No valid split found. Creating leaf node with class {most_common_label}")
            return self.Node(value=most_common_label)

        left_i = X[:, feature] <= threshold
        right_i = X[:, feature] > threshold

        # Print about branching
        if self.verbose:
            print(f"Splitting at feature {feature}, threshold {threshold} -> Left: {np.sum(left_i)} samples, Right: {np.sum(right_i)} samples")

        left = self.build_tree(X[left_i], y[left_i], depth + 1)
        right = self.build_tree(X[right_i], y[right_i], depth + 1)

        return self.Node(feature, threshold, left, right)

    def fit(self, X, y):
        if self.verbose:
            print("Starting to build the tree...")
        self.tree = self.build_tree(X, y)
        if self.verbose:
            print("Finished building the tree.")

    def predict_tree(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_tree(node.left, x)
        else:
            return self.predict_tree(node.right, x)

    def predict(self, X):
        return np.array([self.predict_tree(self.tree, x) for x in X])

    def evaluate(self, X_train, y_train, X_test, y_test):
        if self.verbose:
            print("Evaluating model...")
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        print(f"Accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        disp.plot()
        plt.show(block=True)

        return accuracy
