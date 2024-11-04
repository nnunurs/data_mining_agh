from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decision_tree import DecisionTreeClassifier as OurDecisionTree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
plt.show()

# plt.figure(figsize=(12, 8))
# tree.plot_tree(dt_iris, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
# plt.title("Decision Tree - Iris")
# plt.show()

# #wine


# dt_wine = DecisionTreeClassifier()
# dt_wine.fit(X_train_wine, y_train_wine)

# y_pred_wine = dt_wine.predict(X_test_wine)

# accuracy_wine = accuracy_score(y_test_wine, y_pred_wine)
# results["wine"] = accuracy_wine
# print(f"Wine Accuracy: {accuracy_wine:.2f}")

# plt.figure(figsize=(12, 8))
# tree.plot_tree(dt_wine, filled=True, feature_names=wine.feature_names, class_names=wine.target_names)
# plt.title("Decision Tree - Wine")
# plt.show()
