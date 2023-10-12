from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

import numpy as np


class DPAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_partitions=100, max_depth=None, random_state=None):
        self.n_partitions = n_partitions
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)
        i_shuffle = random_state.permutation(len(X))

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_samples_, self.n_features_in_ = X.shape

        # TODO: generalize to any kind of base estimator

        self.estimators_ = []
        for i, i_sample in enumerate(np.array_split(i_shuffle, self.n_partitions)):
            X_sample = X[i_sample]
            y_sample = y[i_sample]

            self.estimators_.append(
                DecisionTreeClassifier(max_depth=self.max_depth, random_state=i).fit(
                    X_sample, y_sample
                )
            )

    def count_predictions(self, X):
        counts = np.zeros((len(X), self.n_classes_), dtype=int)

        all_indices = np.arange(len(X))
        for estimator in self.estimators_:
            predictions = estimator.predict(X)
            counts[all_indices, predictions] += 1

        return counts

    def compute_poison_robustness(self, X):
        """Returns a lower bound for each sample in X on how many poison samples are required to flip its prediction."""
        counts = self.count_predictions(X)
        class_preds = np.argmax(counts, axis=1)
        most_common_counts = np.max(counts, axis=1)

        counts_corrected = np.copy(counts)
        for i, class_pred in enumerate(class_preds):
            counts_corrected[i, class_pred:] += 1
            counts_corrected[i, class_pred] = 0

        second_most_common_counts = np.max(counts_corrected, axis=1)
        bounds = (most_common_counts - second_most_common_counts) // 2
        return bounds

    def poisoning_accuracy_curve(self, X, y):
        bounds = self.compute_poison_robustness(X)

        predictions = self.predict(X)
        pred_correct = predictions == y

        base_correct = pred_correct.sum()
        correct_counts = np.full(
            shape=self.n_partitions, fill_value=base_correct, dtype=float
        )

        for n_poison_robust, correct in zip(bounds, pred_correct):
            if correct:
                correct_counts[n_poison_robust + 1 :] -= 1

        return correct_counts / len(X)

    def predict(self, X):
        # It is better to predict based on average predicted probabilities
        # but we need to use the majority vote to guarantee poisoning robustness
        return self.classes_.take(np.argmax(self.count_predictions(X), axis=1))
