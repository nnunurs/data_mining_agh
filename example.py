from decision_tree import DecisionTreeClassifier
from sklearn import datasets, model_selection
import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()
X = np.array(iris.data)
y = np.array(iris.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(max_depth=20, criterion='gini')
clf.fit(X_train, y_train)

# Print the tree structure
print("Tree:")
clf.draw_tree()

# Make predictions and evaluate the model
clf.evaluate(X_train, y_train, X_test, y_test)