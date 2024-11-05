# Classifiers and other algorithms in Python built from scratch

This repository contains a custom implementation of a Decision Tree Classifier in Python. The classifier is built from scratch without using any high-level machine learning libraries like `scikit-learn` for the core decision tree logic. The implementation includes methods for training the model, making predictions, visualizing the tree structure, and evaluating the model on different datasets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
  - [DecisionTreeClassifier Class](#decisiontreeclassifier-class)
    - [Methods](#methods)
    - [Example](#example)
    - [Performance](#perf)

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/nnunurs/data_mining_agh.git
cd data_mining_agh
```

2. **Install required libraries**:
```bash
pip install -r requirements.txt
```

## Code Structure

- `decision_tree.py`: The main script containing the implementation of the Decision Tree Classifier.
- `example.py`: A script demonstrating how to use the `DecisionTreeClassifier` with the Iris dataset and evaluate its performance.

### DecisionTreeClassifier Class

The `DecisionTreeClassifier` class encapsulates all the functionality related to the decision tree. It includes methods for calculating entropy and Gini impurity, finding the best split, building the tree, making predictions, visualizing the tree, and evaluating the model.

#### Methods

- `__init__(self, max_depth=10, criterion='gini')`: Initializes the classifier with a maximum depth and a criterion (`gini` or `entropy`).
- `fit(self, X, y)`: Fits the decision tree to the training data.
- `predict(self, X)`: Makes predictions for an array of samples.
- `draw_tree(self, node=None, depth=0)`: Prints the structure of the tree.
- `evaluate(self, X, y, test_size=0.2, random_state=42)`: Evaluates the model on a given dataset by splitting it into training and testing sets, training the model, making predictions, and calculating the accuracy.

#### Example

Run `example.py` to see the Decision Tree Classifier in action. The script loads the Iris dataset, trains the model, makes predictions, visualizes the tree structure, and evaluates the model's performance.

```python
from decision_tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=3, criterion='gini')

# Fit the model to the data
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)

# Visualize the tree structure
clf.draw_tree()

# Evaluate the model
accuracy = clf.evaluate(X, y)
print(f'Model Accuracy: {accuracy:.2f}')
```

This is a simple example to demonstrate how to use the `DecisionTreeClassifier` class. You can experiment with different datasets and hyperparameters to see how the model performs.

#### Performance
**Iris**
![result iris](https://github.com/nnunurs/data_mining_agh/blob/main/results/result_iris.png?raw=true)
**Wine**
![result wine](https://github.com/nnunurs/data_mining_agh/blob/main/results/result_wine.png?raw=true)
**MNIST** (1% subset) 
![result mnist](https://github.com/nnunurs/data_mining_agh/blob/main/results/result_mnist_1%25.png?raw=true)