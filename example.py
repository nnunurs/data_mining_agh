import numpy as np

# Funkcja obliczająca entropię
def entropy(y):
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

# Funkcja obliczająca zysk informacyjny (information gain)
def information_gain(X_column, y, threshold):
    parent_entropy = entropy(y)
    
    # Dzielenie danych na dwie grupy w zależności od progu
    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    
    if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
        return 0

    # Entropia dla lewego i prawego podziału
    n = len(y)
    n_left, n_right = len(y[left_indices]), len(y[right_indices])
    
    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])
    
    # Obliczanie zysku informacyjnego
    weighted_avg_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
    return parent_entropy - weighted_avg_entropy

# Funkcja znajdująca najlepszy podział
def best_split(X, y):
    best_gain = -1
    best_feature = None
    best_threshold = None
    
    # Iterujemy po wszystkich cechach
    for feature_index in range(X.shape[1]):
        X_column = X[:, feature_index]
        thresholds = np.unique(X_column)
        
        # Iterujemy po możliwych progach podziału
        for threshold in thresholds:
            gain = information_gain(X_column, y, threshold)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
    
    return best_feature, best_threshold

# Węzeł drzewa
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Rekurencyjne budowanie drzewa
def build_tree(X, y, depth=0, max_depth=10):
    # Warunek zakończenia
    if len(np.unique(y)) == 1 or depth >= max_depth:
        most_common_label = np.bincount(y).argmax()
        return Node(value=most_common_label)
    
    feature, threshold = best_split(X, y)
    
    if feature is None:
        most_common_label = np.bincount(y).argmax()
        return Node(value=most_common_label)
    
    left_indices = X[:, feature] <= threshold
    right_indices = X[:, feature] > threshold
    
    left = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth)
    right = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth)
    
    return Node(feature, threshold, left, right)

# Funkcja predykcji dla pojedynczego punktu
def predict_single(node, x):
    if node.value is not None:
        return node.value
    
    if x[node.feature] <= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)

# Funkcja predykcji dla zbioru danych
def predict(tree, X):
    return [predict_single(tree, x) for x in X]

# Przykład użycia
if __name__ == "__main__":
    # Przykładowe dane: 6 próbek z 2 cechami
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [3, 2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Budowanie drzewa
    tree = build_tree(X, y, max_depth=3)
    
    # Przewidywanie
    predictions = predict(tree, X)
    print("Predykcje:", predictions)
