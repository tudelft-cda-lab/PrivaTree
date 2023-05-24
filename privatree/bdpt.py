import warnings
from sys import maxsize

import numpy as np
from diffprivlib.mechanisms import Exponential, Laplace, LaplaceTruncated
from diffprivlib.utils import PrivacyLeakWarning
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_random_state, check_X_y

from .privatree import _TREE_LEAF, _TREE_UNDEFINED, CategoricalNode, Node


class BDPTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth=3,
        epsilon=1.0,
        feature_range=None,
        categorical_features=None,
        min_samples_split=2,
        merge_samples=5,
        node_sensitivity="smooth",
        random_state=None,
    ):
        self.max_depth = max_depth
        self.epsilon = epsilon
        self.feature_range = feature_range
        self.categorical_features = categorical_features
        self.min_samples_split = min_samples_split
        self.merge_samples = merge_samples
        self.node_sensitivity = node_sensitivity
        self.random_state = random_state

        if node_sensitivity == "local":
            warnings.warn(
                "Local sensitivity leaks extra information about the dataset. "
                "Use global or smooth sensitivity to prevent leakage.",
                PrivacyLeakWarning,
            )

        half_eps = self.epsilon * 0.5

        self.node_epsilon_ = half_eps / self.max_depth
        self.leaf_epsilon_ = [
            ((1 - np.cbrt(2)) * np.cbrt(2**i))
            / (1 - np.cbrt(2 ** (self.max_depth + 1)))
            * half_eps
            for i in range(self.max_depth + 1)
        ]
        self.random_state_ = check_random_state(self.random_state)

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)

        # We need to copy to merge samples later
        X, y = check_X_y(X, y, copy=True)

        # Feature ranges are ignored (not used) for categorical features
        if self.feature_range is None:
            warnings.warn(
                "Feature ranges have not been specified and will be calculated on the data provided. This will "
                "result in additional privacy leakage. To ensure differential privacy and no additional "
                "privacy leakage, specify bounds for each dimension.",
                PrivacyLeakWarning,
            )

            self.feature_range_ = [
                (X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])
            ]
        else:
            self.feature_range_ = self.feature_range

        if self.categorical_features is not None:
            # TODO: allow string values for categories

            categorical_features = np.array(self.categorical_features)

            if categorical_features.shape[0] != X.shape[1]:
                raise ValueError(
                    f"categorical_features had a different length than n_features {categorical_features.shape[0]} vs {X.shape[1]}"
                )

            if categorical_features.dtype == bool:
                warnings.warn(
                    "Number of categories have not been specified and will be calculated on the data provided. This will "
                    "result in additional privacy leakage. To ensure differential privacy and no additional "
                    "privacy leakage, specify the number of categories (int) for each categorical feature.",
                    PrivacyLeakWarning,
                )

                self.categorical_features_ = []
                for feature_i, is_categorical in enumerate(categorical_features):
                    if not is_categorical:
                        self.categorical_features_.append(0)
                    else:
                        self.categorical_features_.append(
                            int(X[:, feature_i].max() + 1)
                        )
                self.categorical_features_ = np.array(self.categorical_features_)
            elif categorical_features.dtype != int:
                raise ValueError(
                    f"categorical_features needs to be an array of ints or bools, found {categorical_features.dtype}"
                )
            else:
                self.categorical_features_ = categorical_features
        else:
            self.categorical_features_ = np.zeros(X.shape[1], dtype=int)

        if not all(n_cats > 0 for n_cats in self.categorical_features_):
            warnings.warn(
                "BDPT leaks information about the feature values of every group of "
                f"{self.merge_samples} samples for numerical features.",
                PrivacyLeakWarning,
            )

        # Pre-process X such that groups of 5 feature values get merged together into the 1 category
        self.num_feature_to_cat_ = []
        for feature_i in range(X.shape[1]):
            if not self.categorical_features_[feature_i]:
                # If this is a numerical feature
                unique_values, inverse = np.unique(X[:, feature_i], return_inverse=True)
                thresholds = np.empty(len(unique_values) // self.merge_samples + 1)
                cat_i = 0
                for i in range(0, len(unique_values), self.merge_samples):
                    thresholds[cat_i] = unique_values[i]

                    cat_i += 1

                    X[:, feature_i] = inverse // self.merge_samples

                self.num_feature_to_cat_.append(thresholds)
                self.categorical_features_[feature_i] = cat_i
            else:
                self.num_feature_to_cat_.append(None)

        self.root_ = self.__fit_recursive(X, y)

        return self

    def __fit_recursive(self, X, y, depth=0):
        if (
            depth == self.max_depth
            or len(np.unique(y)) == 1
            or len(X) < self.min_samples_split
        ):
            return self.__create_leaf(y, depth)

        feature, threshold, score = self.__find_best_split(X, y, depth)

        # If no split improves the score then we stop and create a leaf
        if threshold is None:
            return self.__create_leaf(y, depth)

        mask_left = X[:, feature] == threshold
        mask_right = np.invert(mask_left)

        X_left = X[mask_left]
        y_left = y[mask_left]
        X_right = X[mask_right]
        y_right = y[mask_right]

        left_node = self.__fit_recursive(X_left, y_left, depth + 1)
        right_node = self.__fit_recursive(X_right, y_right, depth + 1)

        return CategoricalNode(
            feature,
            [threshold],
            left_node,
            right_node,
            _TREE_UNDEFINED,
        )

    def __create_leaf(self, y, depth):
        counts = list(np.bincount(y, minlength=2))

        mech = Laplace(
            epsilon=self.leaf_epsilon_[depth],
            sensitivity=1,
            random_state=self.random_state_,
        )
        count_0 = mech.randomise(counts[0])
        count_1 = mech.randomise(counts[1])

        chosen_label = int(count_1 >= count_0)

        value = np.array([1 - chosen_label, chosen_label])
        return Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)

    def __find_best_split(self, X, y, depth):
        total_count_0, total_count_1 = np.bincount(y, minlength=2)

        mask_0 = y == 0
        mask_1 = ~mask_0

        gini_scores = []
        features = []
        thresholds = []
        for feature_i in range(X.shape[1]):
            l_0 = 0
            l_1 = 0
            r_0 = total_count_0
            r_1 = total_count_1

            feature_values = X[:, feature_i]

            n_categories = self.categorical_features_[feature_i]
            category_values = feature_values.astype(int)
            counts_0 = np.bincount(category_values[mask_0], minlength=n_categories)
            counts_1 = np.bincount(category_values[mask_1], minlength=n_categories)
            for category_i, (count_0, count_1) in enumerate(zip(counts_0, counts_1)):
                r_0 -= count_0
                r_1 -= count_1
                l_0 += count_0
                l_1 += count_1

                if l_0 + l_1 == 0:
                    gini_l = 0
                else:
                    denom = (l_0 + l_1) ** 2
                    gini_l = 1 - (l_0**2) / denom - (l_1**2) / denom

                if r_0 + r_1 == 0:
                    gini_r = 0
                else:
                    denom = (r_0 + r_1) ** 2
                    gini_r = 1 - (r_0**2) / denom - (r_1**2) / denom

                gini = ((l_0 + l_1) * gini_l + (r_0 + r_1) * gini_r) / (
                    l_0 + l_1 + r_0 + r_1
                )

                gini_scores.append(gini)
                features.append(feature_i)
                thresholds.append(category_i)

                r_0 += count_0
                r_1 += count_1
                l_0 -= count_0
                l_1 -= count_1

        if self.node_sensitivity == "local":
            n = len(X)
            sensitivity = 1 - (n / (n + 1)) ** 2 - (1 / (n + 1)) ** 2
        elif self.node_sensitivity == "smooth":
            # Bound the lower by 1 otherwise the sensitivity formula
            # from the paper does not hold.
            mech = LaplaceTruncated(
                epsilon=self.leaf_epsilon_[depth],
                sensitivity=1,
                lower=1,
                upper=maxsize,
                random_state=self.random_state_,
            )
            n = mech.randomise(len(X))
            sensitivity = 1 - (n / (n + 1)) ** 2 - (1 / (n + 1)) ** 2
        elif self.node_sensitivity == "global":
            sensitivity = 0.5
        else:
            raise ValueError(f"Unknown node sensitivity {self.node_sensitivity}")

        # Flip all Gini scores so that we minimize instead of maximize
        utility = [0.5 - score for score in gini_scores]

        mech = Exponential(
            epsilon=self.node_epsilon_,
            sensitivity=sensitivity,
            utility=utility,
            random_state=self.random_state_,
        )
        selected_i = mech.randomise()

        return features[selected_i], thresholds[selected_i], gini_scores[selected_i]

    def predict_proba(self, X):
        # We need to copy to replace numerical values later
        X = check_array(X, copy=True)

        # The BDPT paper does not specify how to apply the sample merging technique
        # to test data so here we map it according to the saved training mapping.
        for feature_i in range(X.shape[1]):
            thresholds = self.num_feature_to_cat_[feature_i]
            if thresholds is not None:
                X[:, feature_i] = (
                    np.searchsorted(thresholds, X[:, feature_i], side="right") - 1
                )

                # If we get a value outside the range map it back to 0
                X[:, feature_i] = np.where(X[:, feature_i] == -1, 0, X[:, feature_i])

        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        return np.array(predictions)

    def predict(self, X):
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=1))
