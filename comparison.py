from sklearn.datasets import load_iris, load_wine, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decision_tree import DecisionTreeClassifier as OurDecisionTree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time

iris = load_iris()
wine = load_wine()
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
subset_size = int(0.01 * len(mnist.data))
indices = np.random.choice(len(mnist.data), subset_size, replace=False)
X_subset, y_subset = mnist.data[indices], mnist.target[indices]

def run_experiment(X_train, X_test, y_train, y_test, dataset_name, max_depth=10):
    results_ours = {}
    results_builtin = {}
    times_ours = {}
    times_builtin = {}

    for depth in range(1, max_depth + 1):
        start_time = time.time()
        dt_ours = OurDecisionTree(max_depth=depth, min_samples_split=10, verbose=True)
        dt_ours.fit(X_train, y_train)
        y_pred_ours = dt_ours.predict(X_test)
        accuracy_ours = accuracy_score(y_test, y_pred_ours)
        results_ours[depth] = accuracy_ours
        times_ours[depth] = time.time() - start_time

        start_time = time.time()
        dt_builtin = DecisionTreeClassifier(max_depth=depth)
        dt_builtin.fit(X_train, y_train)
        y_pred_builtin = dt_builtin.predict(X_test)
        accuracy_builtin = accuracy_score(y_test, y_pred_builtin)
        results_builtin[depth] = accuracy_builtin
        times_builtin[depth] = time.time() - start_time

        print(f"{dataset_name} | Depth: {depth} | Our Model Accuracy: {accuracy_ours:.2f} | Builtin Accuracy: {accuracy_builtin:.2f}")
        print(f"Execution Time - Our Model: {times_ours[depth]:.4f}s | Builtin Model: {times_builtin[depth]:.4f}s")

    return results_ours, results_builtin, times_ours, times_builtin

def plot_results(results_ours, results_builtin, dataset_name):
    plt.plot(list(results_ours.keys()), list(results_ours.values()), label="Ours")
    plt.plot(list(results_builtin.keys()), list(results_builtin.values()), label="Builtin")
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracy")
    plt.title(f"Comparison of Decision Tree Classifiers on {dataset_name} Dataset")
    plt.legend()
    plt.show()

datasets = {
    "Iris": (iris.data, iris.target),
    "Wine": (wine.data, wine.target),
    # "MNIST": (X_subset, y_subset),
}

MAX_DEPTH = 10

for name, (X, y) in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results_ours, results_builtin, times_ours, times_builtin = run_experiment(X_train, X_test, y_train, y_test, name, max_depth=MAX_DEPTH)
    plot_results(results_ours, results_builtin, name)
