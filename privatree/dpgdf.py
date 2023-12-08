import warnings

import numpy as np
from diffprivlib.mechanisms import Exponential, PermuteAndFlip
from diffprivlib.utils import PrivacyLeakWarning
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state, check_X_y

from .privatree import _TREE_LEAF, _TREE_UNDEFINED, CategoricalNode, Node


class DPGDTClassifier(BaseEstimator, ClassifierMixin):

    """
    Main additions of this algorithm compared to plain exponential mechanism
    for splits and leaves:
    - Distributes epsilon 50-50 between leaves and nodes
    - Uses smooth sensitivity for leaf labeling with exponential mechanism
    """

    def __init__(
        self,
        max_depth=3,
        epsilon=1.0,
        feature_range=None,
        categorical_features=None,
        use_smooth_sensitivity_leaves=True,
        node_sensitivity="global",
        min_samples_split=11,
        verbose=False,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.epsilon = epsilon
        self.feature_range = feature_range
        self.categorical_features = categorical_features
        self.use_smooth_sensitivity_leaves = use_smooth_sensitivity_leaves
        self.node_sensitivity = node_sensitivity
        self.min_samples_split = min_samples_split
        self.verbose = verbose
        self.random_state = random_state

        self.node_epsilon_ = self.epsilon / (2 * self.max_depth)
        self.leaf_epsilon_ = self.epsilon / 2
        self.random_state_ = check_random_state(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_, y = np.unique(y, return_inverse=True)

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

            categorical_features = np.asarray(self.categorical_features)

            if len(categorical_features.shape) == 0:
                warnings.warn(
                    "Number of categories have not been specified and will be calculated on the data provided. This will "
                    "result in additional privacy leakage. To ensure differential privacy and no additional "
                    "privacy leakage, specify the number of categories (int) for each categorical feature.",
                    PrivacyLeakWarning,
                )

                self.categorical_features_ = np.array(
                    [int(X[:, feature_i].max() + 1) for feature_i in range(X.shape[1])]
                )
            else:
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
                            raise ValueError(
                                "All features need to be categorical for DPGDF"
                            )
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
                    if any(n_cats == 0 for n_cats in self.categorical_features_):
                        raise ValueError(
                            "All features need to be categorical for DPGDF"
                        )

                    self.categorical_features_ = categorical_features
        else:
            raise ValueError("All features need to be categorical for DPGDF")

        self.root_ = self.__fit_recursive(X, y)

        return self

    def __fit_recursive(self, X, y, depth=0, previous_splits=[]):
        if (
            depth == self.max_depth
        ):
            return self.__create_leaf(y)

        feature, threshold, score = self.__find_best_split(X, y, previous_splits)

        # If no split improves the score then we stop and create a leaf
        if threshold is None:
            return self.__create_leaf(y)

        mask_left = X[:, feature] == threshold
        mask_right = np.invert(mask_left)

        X_left = X[mask_left]
        y_left = y[mask_left]
        X_right = X[mask_right]
        y_right = y[mask_right]

        previous_splits.append((feature, threshold))
        left_node = self.__fit_recursive(X_left, y_left, depth + 1, previous_splits)
        right_node = self.__fit_recursive(X_right, y_right, depth + 1, previous_splits)
        previous_splits.remove((feature, threshold))

        return CategoricalNode(
            feature,
            [threshold],
            left_node,
            right_node,
            _TREE_UNDEFINED,
        )

    def __create_leaf(self, y):
        counts = list(np.bincount(y, minlength=2))

        if self.use_smooth_sensitivity_leaves:
            second_majority, majority = np.sort(counts)[-2:]
            utility = [float(count == majority) for count in counts]

            difference = majority - second_majority
            smooth_sensitivity = np.exp(-difference * self.leaf_epsilon_)

            if self.verbose:
                print("smooth sensitivity:", smooth_sensitivity)

            mech = Exponential(
                epsilon=self.leaf_epsilon_,
                sensitivity=smooth_sensitivity,
                monotonic=True,
                utility=utility,
                random_state=self.random_state_,
            )
            chosen_label = mech.randomise()
        else:
            mech = PermuteAndFlip(
                epsilon=self.leaf_epsilon_,
                sensitivity=1,
                monotonic=True,
                utility=counts,
                random_state=self.random_state_,
            )
            chosen_label = mech.randomise()

        if self.verbose:
            if chosen_label != np.argmax(counts):
                print(f"Leaf with counts={counts} gets wrong label ({chosen_label})")
            else:
                print(f"Leaf with counts={counts} gets correct label ({chosen_label})")

        value = np.array([1 - chosen_label, chosen_label])
        return Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)

    def __find_best_split(self, X, y, previous_splits):
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
            if n_categories:
                category_values = feature_values.astype(int)
                counts_0 = np.bincount(category_values[mask_0], minlength=n_categories)
                counts_1 = np.bincount(category_values[mask_1], minlength=n_categories)
                for category_i, (count_0, count_1) in enumerate(
                    zip(counts_0, counts_1)
                ):
                    # Ignore attributes that we already used since these
                    # are useless anyway.
                    if (feature_i, category_i) in previous_splits:
                        continue

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

                    if l_0 + l_1 + r_0 + r_1 == 0:
                        gini = 0.5
                    else:
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
            else:
                raise ValueError("Every feature should be categorical")

        if len(gini_scores) == 0:
            return None, None, None

        if self.node_sensitivity == "local":
            # Compute the sensitivity based on the number of available samples
            n = len(X)
            sensitivity = 1 - (n / (n + 1)) ** 2 - (1 / (n + 1)) ** 2
        elif self.node_sensitivity == "global":
            sensitivity = 0.5
        else:
            raise ValueError(f"Unknown node sensitivity {self.node_sensitivity}")

        # Flip all Gini scores so that we minimize instead of maximize
        utility = [0.5 - score for score in gini_scores]

        mech = PermuteAndFlip(
            epsilon=self.node_epsilon_,
            sensitivity=sensitivity,
            utility=utility,
            random_state=self.random_state_,
        )
        selected_i = mech.randomise()

        if self.verbose:
            print(f"selected utility: {utility[selected_i]}")

        return features[selected_i], thresholds[selected_i], gini_scores[selected_i]

    def predict_proba(self, X):
        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        return np.array(predictions)

    def predict(self, X):
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=1))
