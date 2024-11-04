from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decision_tree import DecisionTreeClassifier as OurDecisionTree
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2
)

wine = load_wine()
X_wine, y_wine = wine.data, wine.target
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.2
)

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_mnist, y_mnist = mnist.data, mnist.target
subset_size = int(0.01 * len(X_mnist))  # 10% z całego zbioru
indices = np.random.choice(len(X_mnist), subset_size, replace=False)
X_subset, y_subset = X_mnist[indices], y_mnist[indices]
X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)

results_ours = {}
results_builtin = {}


def run(depth, model, X_train, X_test, y_train, y_test):
    dt_iris = model(max_depth=depth)
    dt_iris.fit(X_train, y_train)

    y_pred = dt_iris.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


MAX_DEPTH = 10

for depth in range(1, MAX_DEPTH):
    print(f"Depth: {depth}")
    results_ours[depth] = run(
        depth, OurDecisionTree, X_train_iris, X_test_iris, y_train_iris, y_test_iris
    )
    results_builtin[depth] = run(
        depth,
        DecisionTreeClassifier,
        X_train_iris,
        X_test_iris,
        y_train_iris,
        y_test_iris,
    )

plt.plot(list(results_ours.keys()), list(results_ours.values()), label="Ours")
plt.plot(list(results_builtin.keys()), list(results_builtin.values()), label="Builtin")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Comparison of Decision Tree Classifiers on Iris Dataset")
plt.legend()
plt.show()

results_ours = {}
results_builtin = {}

for depth in range(1, MAX_DEPTH):
    print(f"Depth: {depth}")
    results_ours[depth] = run(
        depth, OurDecisionTree, X_train_wine, X_test_wine, y_train_wine, y_test_wine
    )
    results_builtin[depth] = run(
        depth,
        DecisionTreeClassifier,
        X_train_wine,
        X_test_wine,
        y_train_wine,
        y_test_wine,
    )

plt.plot(list(results_ours.keys()), list(results_ours.values()), label="Ours")
plt.plot(list(results_builtin.keys()), list(results_builtin.values()), label="Builtin")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Comparison of Decision Tree Classifiers on Wine Dataset")
plt.legend()
plt.show()



results_ours_mnist = {}
results_builtin_mnist = {}

for depth in range(1, MAX_DEPTH):
    print(f"Depth: {depth}")
    results_ours_mnist[depth] = run(
        depth, OurDecisionTree, X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist
    )
    results_builtin_mnist[depth] = run(
        depth,
        DecisionTreeClassifier,
        X_train_mnist,
        X_test_mnist,
        y_train_mnist,
        y_test_mnist,
    )

plt.plot(list(results_ours_mnist.keys()), list(results_ours_mnist.values()), label="Ours")
plt.plot(list(results_builtin_mnist.keys()), list(results_builtin_mnist.values()), label="Builtin")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Comparison of Decision Tree Classifiers on MNIST Dataset")
plt.legend()
plt.show()