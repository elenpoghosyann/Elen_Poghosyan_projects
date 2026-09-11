

import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# TreeNode
# ---------------------------------------------------------------------------

class TreeNode:
 

    def __init__(self):
        self.feature   = None
        self.threshold = None
        self.left      = None
        self.right     = None
        self.is_leaf   = False
        self.value     = None
        self.samples   = None
        self.impurity  = None


# ---------------------------------------------------------------------------
# DecisionTreeClassifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
 
    def __init__(
        self,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        if criterion not in ('gini', 'entropy'):
            raise ValueError("criterion must be 'gini' or 'entropy'")

        self.criterion         = criterion
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.random_state      = random_state

        self.root_              = None
        self.n_features_        = None
        self.n_classes_         = None
        self.classes_           = None
        self.feature_importances_ = None
        self._rng               = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
     
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_    = np.unique(y)
        self.n_classes_  = len(self.classes_)
        self.n_features_ = X.shape[1]

        # Accumulator for feature importances (weighted impurity decrease)
        self._importance_acc = np.zeros(self.n_features_, dtype=float)

        self.root_ = self._build_tree(X, y, depth=0)

        # Normalise importances
        total = self._importance_acc.sum()
        if total > 0:
            self.feature_importances_ = self._importance_acc / total
        else:
            self.feature_importances_ = self._importance_acc.copy()

        return self

    def predict(self, X):
   
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_sample(x, self.root_) for x in X])

    def predict_proba(self, X):
        
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_proba_sample(x, self.root_) for x in X])

    def get_feature_importance(self):
      
        if self.feature_importances_ is None:
            raise RuntimeError("Call fit() before get_feature_importance().")
        return self.feature_importances_

    # ------------------------------------------------------------------
    # Impurity helpers
    # ------------------------------------------------------------------

    def _gini(self, y):
    
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs  = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _entropy(self, y):
      
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs  = counts[counts > 0] / len(y)
        return -np.sum(probs * np.log2(probs))

    def _impurity(self, y):
      
        if self.criterion == 'gini':
            return self._gini(y)
        return self._entropy(y)

    def _information_gain(self, y, y_left, y_right):
       
        n       = len(y)
        n_left  = len(y_left)
        n_right = len(y_right)

        if n_left == 0 or n_right == 0:
            return 0.0

        parent_impurity = self._impurity(y)
        child_impurity  = (
            (n_left  / n) * self._impurity(y_left) +
            (n_right / n) * self._impurity(y_right)
        )
        return parent_impurity - child_impurity

    # ------------------------------------------------------------------
    # Split search
    # ------------------------------------------------------------------

    def _get_feature_indices(self, n_features):
      
        if self.max_features is None:
            return np.arange(n_features)

        if self.max_features == 'sqrt':
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            k = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
        else:
            return np.arange(n_features)

        return self._rng.choice(n_features, size=k, replace=False)

    def _find_best_split(self, X, y):
     
        best_feature   = None
        best_threshold = None
        best_gain      = 0.0

        feature_indices = self._get_feature_indices(X.shape[1])

        for feat_idx in feature_indices:
            feature_values = X[:, feat_idx]
            # Use midpoints between sorted unique values as candidate thresholds
            unique_vals = np.unique(feature_values)
            thresholds  = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask  = feature_values <= threshold
                right_mask = ~left_mask

                y_left  = y[left_mask]
                y_right = y[right_mask]

                # Skip splits that would violate min_samples_leaf
                if (len(y_left)  < self.min_samples_leaf or
                        len(y_right) < self.min_samples_leaf):
                    continue

                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feat_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _stopping_condition(self, y, depth):
      
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        if len(np.unique(y)) == 1:
            return True
        return False

    def _majority_class(self, y):
     
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _build_tree(self, X, y, depth):
        
        node          = TreeNode()
        node.samples  = len(y)
        node.impurity = self._impurity(y)

        # --- Stopping condition: create leaf ---
        if self._stopping_condition(y, depth):
            node.is_leaf = True
            node.value   = self._majority_class(y)
            return node

        # --- Find best split ---
        feat_idx, threshold, gain = self._find_best_split(X, y)

        # No informative split found → leaf
        if feat_idx is None or gain == 0.0:
            node.is_leaf = True
            node.value   = self._majority_class(y)
            return node

        # --- Record weighted impurity decrease for feature importance ---
        n_total = len(y)
        left_mask  = X[:, feat_idx] <= threshold
        right_mask = ~left_mask
        X_left,  y_left  = X[left_mask],  y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        weighted_decrease = (
            node.impurity
            - (len(y_left)  / n_total) * self._impurity(y_left)
            - (len(y_right) / n_total) * self._impurity(y_right)
        )
        self._importance_acc[feat_idx] += weighted_decrease * n_total

        # --- Create internal node and recurse ---
        node.feature   = feat_idx
        node.threshold = threshold
        node.left      = self._build_tree(X_left,  y_left,  depth + 1)
        node.right     = self._build_tree(X_right, y_right, depth + 1)

        return node

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_sample(self, x, node):
       
        if node.is_leaf:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def _predict_proba_sample(self, x, node):
        
        if node.is_leaf:
            proba = np.zeros(self.n_classes_)
            class_idx = np.where(self.classes_ == node.value)[0][0]
            proba[class_idx] = 1.0
            return proba

        if x[node.feature] <= node.threshold:
            return self._predict_proba_sample(x, node.left)
        return self._predict_proba_sample(x, node.right)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_depth(self):
        
        return self._node_depth(self.root_)

    def _node_depth(self, node):
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._node_depth(node.left), self._node_depth(node.right))

    def get_n_leaves(self):
       
        return self._count_leaves(self.root_)

    def _count_leaves(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def __repr__(self):
        return (
            f"DecisionTreeClassifier("
            f"criterion='{self.criterion}', "
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split})"
        )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for criterion in ('gini', 'entropy'):
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=5, random_state=42)
        tree.fit(X_train, y_train)
        train_acc = (tree.predict(X_train) == y_train).mean()
        test_acc  = (tree.predict(X_test)  == y_test ).mean()
        print(f"[Iris | {criterion:7s}]  "
              f"train={train_acc:.4f}  test={test_acc:.4f}  "
              f"depth={tree.get_depth()}  leaves={tree.get_n_leaves()}")
