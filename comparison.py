from sklearn.datasets import load_iris, load_wine, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decision_tree import DecisionTreeClassifier as OurDecisionTree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Ładowanie danych
iris = load_iris()
wine = load_wine()
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
# Użycie 10% MNIST dla optymalizacji
subset_size = int(0.01 * len(mnist.data))
indices = np.random.choice(len(mnist.data), subset_size, replace=False)
X_subset, y_subset = mnist.data[indices], mnist.target[indices]

# Funkcja do trenowania modelu i ewaluacji
def run_experiment(X_train, X_test, y_train, y_test, dataset_name, max_depth=10):
    results_ours = {}
    results_builtin = {}

    for depth in range(1, max_depth + 1):
        # Nasz model
        dt_ours = OurDecisionTree(max_depth=depth, min_samples_split=10, verbose=True)
        dt_ours.fit(X_train, y_train)
        y_pred_ours = dt_ours.predict(X_test)
        accuracy_ours = accuracy_score(y_test, y_pred_ours)
        results_ours[depth] = accuracy_ours

        # Wbudowany model
        dt_builtin = DecisionTreeClassifier(max_depth=depth)
        dt_builtin.fit(X_train, y_train)
        y_pred_builtin = dt_builtin.predict(X_test)
        accuracy_builtin = accuracy_score(y_test, y_pred_builtin)
        results_builtin[depth] = accuracy_builtin

        # Wypisywanie wyników po każdej iteracji
        print(f"{dataset_name} | Depth: {depth} | Our Model Accuracy: {accuracy_ours:.2f} | Builtin Accuracy: {accuracy_builtin:.2f}")

    return results_ours, results_builtin

# Funkcja do rysowania wykresu
def plot_results(results_ours, results_builtin, dataset_name):
    plt.plot(list(results_ours.keys()), list(results_ours.values()), label="Ours")
    plt.plot(list(results_builtin.keys()), list(results_builtin.values()), label="Builtin")
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracy")
    plt.title(f"Comparison of Decision Tree Classifiers on {dataset_name} Dataset")
    plt.legend()
    plt.show()

# Eksperymenty na zestawach danych
datasets = {
    "Iris": (iris.data, iris.target),
    "Wine": (wine.data, wine.target),
    "MNIST": (X_subset, y_subset),
}

# Głębokość drzewa do przetestowania
MAX_DEPTH = 10

# Uruchomienie eksperymentów i rysowanie wykresów
for name, (X, y) in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results_ours, results_builtin = run_experiment(X_train, X_test, y_train, y_test, name, max_depth=MAX_DEPTH)
    plot_results(results_ours, results_builtin, name)
