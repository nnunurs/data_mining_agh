# Classifiers and other algorithms in Python built from scratch

This repository contains a custom implementation of a Decision Tree Classifier in Python. The classifier is built from scratch without using any high-level machine learning libraries like `scikit-learn` for the core decision tree logic. The implementation includes methods for training the model, making predictions, visualizing the tree structure, and evaluating the model on different datasets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
  - [DecisionTreeClassifier Class](#decisiontreeclassifier-class)
    - [Methods](#methods)
    - [Example](#example)

## Installation

To run the code, you need to have Python installed along with the following libraries:

- `numpy`
- `scikit-learn`

You can install the required libraries using pip:

```bash
pip install numpy scikit-learn
```

## Usage

1. **Clone the repository**:

```bash
git clone https://github.com/nnunurs/modelowanie.git
cd modelowanie
```

2. **Check the example**:
```bash
python example.py
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