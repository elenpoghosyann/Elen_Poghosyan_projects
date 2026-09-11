

import numpy as np
from collections import Counter
from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
 
    def __init__(
        self,
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        oob_score=False,
        random_state=None,
    ):
        self.n_estimators      = n_estimators
        self.criterion         = criterion
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.bootstrap         = bootstrap
        self.oob_score         = oob_score
        self.random_state      = random_state

        self.estimators_          = []       # trained trees
        self.oob_score_           = None     # OOB accuracy (if requested)
        self.feature_importances_ = None     # averaged importances
        self.classes_             = None
        self.n_classes_           = None
        self.n_features_          = None
        self._rng                 = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
     
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_samples, n_features    = X.shape
        self.classes_            = np.unique(y)
        self.n_classes_          = len(self.classes_)
        self.n_features_         = n_features
        self.estimators_         = []

        # OOB accumulators: votes[i] = Counter of predictions for sample i
        if self.oob_score:
            oob_votes = [Counter() for _ in range(n_samples)]

        importance_acc = np.zeros(n_features, dtype=float)

        for i in range(self.n_estimators):
            # ---- 1. Bootstrap sample --------------------------------
            if self.bootstrap:
                indices     = self._bootstrap_sample(n_samples)
                X_boot      = X[indices]
                y_boot      = y[indices]
            else:
                indices     = np.arange(n_samples)
                X_boot      = X
                y_boot      = y

            # ---- 2. Train one tree with feature randomness ----------
            seed = int(self._rng.integers(0, 2**31))
            tree = DecisionTreeClassifier(
                criterion         = self.criterion,
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf  = self.min_samples_leaf,
                max_features      = self.max_features,
                random_state      = seed,
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

            # ---- 3. Accumulate feature importances ------------------
            importance_acc += tree.feature_importances_

            # ---- 4. OOB predictions ---------------------------------
            if self.oob_score and self.bootstrap:
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[np.unique(indices)] = False
                oob_indices = np.where(oob_mask)[0]
                if len(oob_indices) > 0:
                    preds = tree.predict(X[oob_indices])
                    for idx, pred in zip(oob_indices, preds):
                        oob_votes[idx][pred] += 1

        # ---- Normalise feature importances --------------------------
        total = importance_acc.sum()
        self.feature_importances_ = (
            importance_acc / total if total > 0 else importance_acc
        )

        # ---- Compute OOB accuracy -----------------------------------
        if self.oob_score and self.bootstrap:
            self.oob_score_ = self._compute_oob_accuracy(y, oob_votes, n_samples)

        return self

    def predict(self, X):
       
        X = np.asarray(X, dtype=float)
        # Collect predictions from every tree: shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        # Majority vote for each sample
        y_pred = np.apply_along_axis(
            lambda col: Counter(col).most_common(1)[0][0],
            axis=0,
            arr=all_preds,
        )
        return y_pred

    def predict_proba(self, X):
        
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        proba_sum = np.zeros((n_samples, self.n_classes_), dtype=float)

        for tree in self.estimators_:
            preds = tree.predict(X)
            for i, pred in enumerate(preds):
                class_idx = np.where(self.classes_ == pred)[0][0]
                proba_sum[i, class_idx] += 1.0

        return proba_sum / self.n_estimators

    def get_feature_importance(self):
       
        if self.feature_importances_ is None:
            raise RuntimeError("Call fit() before get_feature_importance().")
        return self.feature_importances_

    # ------------------------------------------------------------------
    # Bootstrap helper
    # ------------------------------------------------------------------

    def _bootstrap_sample(self, n_samples):
      
        return self._rng.integers(0, n_samples, size=n_samples)

    # ------------------------------------------------------------------
    # OOB helper
    # ------------------------------------------------------------------

    def _compute_oob_accuracy(self, y, oob_votes, n_samples):
    
        correct = 0
        evaluated = 0
        for i in range(n_samples):
            if len(oob_votes[i]) == 0:
                continue   # sample was never OOB (rare with many trees)
            oob_pred = oob_votes[i].most_common(1)[0][0]
            if oob_pred == y[i]:
                correct += 1
            evaluated += 1

        return correct / evaluated if evaluated > 0 else 0.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"RandomForestClassifier("
            f"n_estimators={self.n_estimators}, "
            f"criterion='{self.criterion}', "
            f"max_depth={self.max_depth}, "
            f"max_features={self.max_features!r})"
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

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        max_features='sqrt',
        oob_score=True,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    train_acc = (rf.predict(X_train) == y_train).mean()
    test_acc  = (rf.predict(X_test)  == y_test ).mean()

    print(f"[Iris | Random Forest]  "
          f"train={train_acc:.4f}  test={test_acc:.4f}  "
          f"OOB={rf.oob_score_:.4f}")
    print(f"Top feature importances: {rf.feature_importances_.round(3)}")
