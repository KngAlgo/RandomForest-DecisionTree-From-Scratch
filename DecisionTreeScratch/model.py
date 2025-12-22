import numpy as np

def gini_impurity(input):
    if len(input) == 0:
        return 0
    else:
        array = np.array(input)
        n_count = np.bincount(input)
        n_total = len(input)
        proportions = n_count / n_total
        return 1 - sum(proportions**2)

def information_gain(parent_labels, left_labels, right_labels):
    parent_gini = gini_impurity(parent_labels)
    left_gini = gini_impurity(left_labels)
    right_gini = gini_impurity(right_labels)
    n_total = len(left_labels) + len(right_labels)
    weighted_gini = (len(left_labels)/n_total * left_gini) + (len(right_labels)/n_total * right_gini)
    return parent_gini - weighted_gini

def find_best_split(X, y, max_features=None):
    # X is a 2D numpy array (rows = samples, columns = features)
    # y is a 1D numpy array of labels
    
    best_gain = -1
    best_feature = None
    best_threshold = None

    if max_features is None:
        feature_indices = range(X.shape[1])
    else:
        feature_indices = np.random.choice(range(X.shape[1]), size=max_features, replace=False)
    
    # Loop through each feature (column)
    for feature_idx in feature_indices:
        # Get all unique values in this feature as potential thresholds
        thresholds = np.unique(X[:, feature_idx])
        
        # Try each threshold
        for threshold in thresholds:
            # Split data: left is where feature <= threshold
            left_mask = X[:, feature_idx] <= threshold

            left_labels = y[left_mask]
            right_labels = y[~left_mask]

            # Calculate information gain
            gain = information_gain(y, left_labels, right_labels)
            
            # Update best split if this is better
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold


class Node():
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx # which feature to split on
        self.threshold = threshold # value to compare against
        self.left = left # pointer to left child node
        self.right = right # pointer to right child node
        self.value = value # Leaf node attribute, prediction
    
    def is_leaf(self):
        return self.value is not None


class DecisionTree():
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth # Max depth of tree
        self.min_samples_split = min_samples_split # min samples needed to split
        self.root = None # Root Node, gets value after fitting
        self.max_features = max_features

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y) # get n samples
        n_classes = len(np.unique(y)) # get num classes

        # Return Leaf Node if pure node, reached max depth, or too few samples to split

        if (n_classes == 1 or (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split):
            leaf_value = np.bincount(y).argmax() # finds number of majority class
            return Node(value=leaf_value) # returns a leaf node of the classes
        
        best_feature, best_threshold = find_best_split(X, y, max_features=self.max_features) # finds the best feature and threshold using our data and find_best_split function

        # If no good split found, return leaf
        if best_feature is None:
            leaf_value = np.bincount(y).argmax() # finds majority class
            return Node(value=leaf_value)
        
        # creates a left mask by splitting the data based on the best feature under the best threshold
        left_mask = X[:, best_feature] <= best_threshold 
        x_left, y_left = X[left_mask], y[left_mask] # associate left and right groupings
        x_right, y_right = X[~left_mask], y[~left_mask]

        left_child = self._build_tree(x_left, y_left, depth+1) # implement recursion
        right_child = self._build_tree(x_right, y_right, depth+1)

        return Node(feature_idx=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def predict(self, X):
        # Loops through all samples and calls _predict_sample on them.
        # X is 2D with columns and rows
        preds = [self._predict_sample(sample, self.root) for sample in X]
        return np.array(preds)
        

    def _predict_sample(self, sample, node):
        # If we're at leaf node, return node value
        if node.is_leaf():
            return node.value
        
        # Same sort of recursion we saw earlier
        if sample[node.feature_idx] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)

class RandomForest():
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        
    def fit(self, X, y):
        for i in range(self.n_trees):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            x_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        preds = []
        for i in self.trees:
            pred = i.predict(X)
            preds.append(pred)

        preds = np.array(preds).T

        final_preds = []
        for sample_pred in preds:
            maj_vote = np.bincount(sample_pred).argmax()
            final_preds.append(maj_vote)
        
        return np.array(final_preds)
            



if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create a simple dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=7, 
                               n_redundant=2, random_state=42)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the tree
    tree = DecisionTree(max_depth=5)
    tree.fit(X_train, y_train)
    
    # Make predictions
    predictions = tree.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Decision Tree Accuracy: {accuracy:.2f}")
    print(f"Predictions: {predictions[:10]}")
    print(f"Actual:      {y_test[:10]}")

# Train the Random Forest
print("\n" + "="*50)
forest = RandomForest(n_trees=10, max_depth=5, max_features=3)
forest.fit(X_train, y_train)

# Make predictions
rf_predictions = forest.predict(X_test)

# Calculate accuracy
rf_accuracy = np.mean(rf_predictions == y_test)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"Predictions: {rf_predictions[:10]}")
print(f"Actual:      {y_test[:10]}")

print("\n" + "="*50)
print(f"Improvement: {rf_accuracy - accuracy:.2f}")