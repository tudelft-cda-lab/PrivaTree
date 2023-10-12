import warnings
from collections import defaultdict
from sys import maxsize

import numpy as np
from diffprivlib.mechanisms import GeometricTruncated, PermuteAndFlip
from diffprivlib.utils import PrivacyLeakWarning
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted

from .joint_exp import private_quantiles

_TREE_LEAF = -1
_TREE_UNDEFINED = -2


class Node:
    """Base class for decision tree nodes, also functions as leaf."""

    def __init__(self, feature, left_child, right_child, value):
        self.feature = feature
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

    def predict(self, _):
        return self.value

    def to_xgboost_json(self, node_id, depth):
        if isinstance(self.value, np.ndarray):
            # Return leaf value in range [-1, 1]
            return {"nodeid": node_id, "leaf": self.value[1] * 2 - 1}, node_id
        else:
            return {"nodeid": node_id, "leaf": self.value}, node_id

    def is_leaf(self):
        return self.left_child == _TREE_LEAF and self.right_child == _TREE_LEAF

    def prune(self, _):
        return self


class NumericalNode(Node):
    """
    Decision tree node for numerical decision (threshold).
    """

    def __init__(self, feature, threshold, left_child, right_child, value):
        super().__init__(feature, left_child, right_child, value)
        self.threshold = threshold

    def predict(self, sample):
        """
        Predict the class label of the given sample. Follow the left subtree
        if the sample's value is lower or equal to the threshold, else follow
        the right sub tree.
        """
        comparison = sample[self.feature] <= self.threshold
        if comparison:
            return self.left_child.predict(sample)
        else:
            return self.right_child.predict(sample)

    def to_xgboost_json(self, node_id, depth):
        left_id = node_id + 1
        left_dict, new_node_id = self.left_child.to_xgboost_json(left_id, depth + 1)

        right_id = new_node_id + 1
        right_dict, new_node_id = self.right_child.to_xgboost_json(right_id, depth + 1)

        return (
            {
                "nodeid": node_id,
                "depth": depth,
                "split": self.feature,
                "split_condition": self.threshold,
                "yes": left_id,
                "no": right_id,
                "missing": left_id,
                "children": [left_dict, right_dict],
            },
            new_node_id,
        )

    def prune(self, bounds=defaultdict(lambda: [-np.inf, np.inf])):
        old_high = bounds[self.feature][1]
        bounds[self.feature][1] = self.threshold

        self.left_child = self.left_child.prune(bounds)

        bounds[self.feature][1] = old_high
        old_low = bounds[self.feature][0]
        bounds[self.feature][0] = self.threshold

        self.right_child = self.right_child.prune(bounds)

        bounds[self.feature][0] = old_low

        if self.threshold >= bounds[self.feature][1] or self.threshold == np.inf:
            # If no sample can reach this node's right side
            return self.left_child
        elif self.threshold <= bounds[self.feature][0] or self.threshold == -np.inf:
            # If no sample can reach this node's left side
            return self.right_child
        elif (
            self.left_child.is_leaf()
            and self.right_child.is_leaf()
            and self.left_child.value[1] == self.right_child.value[1]
        ):
            # If both children are leaves and they predict the same value
            return self.left_child
        else:
            return self


class CategoricalNode(Node):
    """
    Decision tree node for numerical decision (threshold).
    """

    def __init__(self, feature, categories, left_child, right_child, value):
        super().__init__(feature, left_child, right_child, value)
        self.categories = categories

    def predict(self, sample):
        """
        Predict the class label of the given sample. Follow the left subtree
        if the sample's categorical value is in the list of left categories
        else go to the right.
        """
        comparison = sample[self.feature] in self.categories
        if comparison:
            return self.left_child.predict(sample)
        else:
            return self.right_child.predict(sample)

    def to_xgboost_json(self, node_id, depth):
        left_id = node_id + 1
        left_dict, new_node_id = self.left_child.to_xgboost_json(left_id, depth + 1)

        right_id = new_node_id + 1
        right_dict, new_node_id = self.right_child.to_xgboost_json(right_id, depth + 1)

        return (
            {
                "nodeid": node_id,
                "depth": depth,
                "split": self.feature,
                "categories": self.categories,
                "yes": left_id,
                "no": right_id,
                "missing": left_id,
                "children": [left_dict, right_dict],
            },
            new_node_id,
        )

    def prune(self, bounds=defaultdict(lambda: [-np.inf, np.inf])):
        self.left_child = self.left_child.prune(bounds)
        self.right_child = self.right_child.prune(bounds)

        if len(self.categories) == 0:
            return self.right_child
        elif (
            self.left_child.is_leaf()
            and self.right_child.is_leaf()
            and self.left_child.value[1] == self.right_child.value[1]
        ):
            # If both children are leaves and they predict the same value
            return self.left_child
        else:
            return self


def private_bincount(sample, n_bins, epsilon, random_state):
    hist = np.bincount(sample, minlength=n_bins)

    dp_mech = GeometricTruncated(
        epsilon=epsilon,
        sensitivity=1,
        lower=0,
        upper=maxsize,
        random_state=random_state,
    )
    dp_hist = np.zeros_like(hist)

    for i in np.arange(dp_hist.shape[0]):
        dp_hist[i] = dp_mech.randomise(int(hist[i]))

    return dp_hist


def _worst_expected_error_pf(n_classes):
    def neg_f(p):
        # This is the negative of the expression that describes the worst-case
        # expected error of permute-and-flip for n candidates.
        # Here delta=1 because adding 1 item to the database increases
        # sample counts by 1.
        return -2 * np.log(1 / p) * (1 - (1 - (1 - p) ** n_classes) / (n_classes * p))

    # we take the negative because we want to maximize
    return -minimize_scalar(neg_f, bounds=(0, 1)).fun


class PrivateBinner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_bins=10,
        epsilon=1.0,
        feature_range=None,
        categorical_features=None,
        use_private_quantiles=True,
        random_state=None,
    ):
        self.max_bins = max_bins
        self.epsilon = epsilon
        self.feature_range = feature_range
        self.categorical_features = categorical_features
        self.use_private_quantiles = use_private_quantiles
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.random_state_ = check_random_state(self.random_state)

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

        quantiles = np.linspace(0, 1, self.max_bins + 1)

        self.feature_to_bin_info_ = []
        self.n_bins_ = []
        for feature_i in range(X.shape[1]):
            n_categories = self.categorical_features_[feature_i]
            if n_categories:
                self.feature_to_bin_info_.append(None)
                self.n_bins_.append(n_categories)
            else:
                if self.use_private_quantiles:
                    bins = np.unique(
                        list(
                            private_quantiles(
                                sorted_data=sorted(X[:, feature_i]),
                                data_low=self.feature_range_[feature_i][0],
                                data_high=self.feature_range_[feature_i][1],
                                qs=quantiles[1:-1],
                                eps=self.epsilon,
                                random_state=self.random_state_,
                            )
                        )
                        + [self.feature_range_[feature_i][1]]
                    )
                    self.feature_to_bin_info_.append(bins)
                    self.n_bins_.append(len(bins))
                else:
                    bins = np.unique(
                        list(
                            np.quantile(
                                X[:, feature_i],
                                quantiles[1:-1],
                            )
                        )
                        + [self.feature_range_[feature_i][1]]
                    )
                    self.feature_to_bin_info_.append(bins)
                    self.n_bins_.append(len(bins))

        return self

    def transform(self, X):
        X = check_array(X)
        check_is_fitted(self, "n_bins_")

        X_binned = np.empty(X.shape, dtype=np.uint8)

        for feature_i, values in enumerate(X.T):
            if self.categorical_features_[feature_i]:
                # Categorical variables are already mapped to ints
                X_binned[:, feature_i] = X[:, feature_i]
            else:
                # For numerical features we get a vector of bins
                # that we can use to digitize.
                bins = self.feature_to_bin_info_[feature_i]
                X_binned[:, feature_i] = np.digitize(values, bins, right=True)

        return X_binned

    def bin_to_node(self, feature, split_bin, bin_order=None):
        check_is_fitted(self, "n_bins_")

        if self.categorical_features_[feature]:
            if bin_order is None:
                raise ValueError(
                    "bin_order needs to be supplied for categorical splits"
                )

            return CategoricalNode(
                feature=feature,
                categories=list(bin_order[: split_bin + 1]),
                left_child=_TREE_UNDEFINED,
                right_child=_TREE_UNDEFINED,
                value=_TREE_UNDEFINED,
            )
        else:
            bins = self.feature_to_bin_info_[feature]

            return NumericalNode(
                feature=feature,
                threshold=bins[split_bin],
                left_child=_TREE_UNDEFINED,
                right_child=_TREE_UNDEFINED,
                value=_TREE_UNDEFINED,
            )


class PrivaTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth=3,
        epsilon=1.0,
        max_bins=10,
        feature_range=None,
        categorical_features=None,
        min_samples_split=2,
        min_samples_leaf=1,
        epsilon_shares="auto",
        leaf_label_success_prob=0.99,
        max_leaf_epsilon_share=0.5,
        use_private_quantiles=True,
        verbose=False,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.epsilon = epsilon
        self.max_bins = max_bins
        self.feature_range = feature_range
        self.categorical_features = categorical_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.epsilon_shares = epsilon_shares
        self.leaf_label_success_prob = leaf_label_success_prob
        self.max_leaf_epsilon_share = max_leaf_epsilon_share
        self.use_private_quantiles = use_private_quantiles
        self.verbose = verbose
        self.random_state = random_state

        if self.max_bins > 255:
            raise ValueError(f"n_bins cannot be larger than 255 ({self.max_bins})")

        self.random_state_ = check_random_state(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_samples_, self.n_features_in_ = X.shape

        self.__distribute_epsilon()

        self.binner_ = PrivateBinner(
            max_bins=self.max_bins,
            epsilon=self.quantile_epsilon_,
            feature_range=self.feature_range,
            categorical_features=self.categorical_features,
            use_private_quantiles=self.use_private_quantiles,
            random_state=self.random_state_,
        )
        X_binned = self.binner_.fit_transform(X, y)
        self.n_bins_ = self.binner_.n_bins_

        # Do the actual training procedure by recursively splitting
        self.root_ = self.__fit_recursive(X_binned, y)

        return self

    def __distribute_epsilon(self):
        if self.epsilon_shares == "auto":
            required_leaf_epsilon = (
                _worst_expected_error_pf(len(self.classes_)) * (2**self.max_depth)
            ) / (self.n_samples_ * (1 - self.leaf_label_success_prob))

            max_leaf_epsilon = self.epsilon * self.max_leaf_epsilon_share
            self.leaf_epsilon_ = min(max_leaf_epsilon, required_leaf_epsilon)

            if self.verbose:
                print("required leaf epsilon:", required_leaf_epsilon)
                print("maximum leaf epsilon:", max_leaf_epsilon)
                print("total epsilon budget:", self.epsilon)

            if self.use_private_quantiles:
                self.quantile_epsilon_ = (self.epsilon - self.leaf_epsilon_) / (
                    self.max_depth + 1
                )
                self.node_num_epsilon_ = (self.epsilon - self.leaf_epsilon_) / (
                    self.max_depth + 1
                )
                self.node_cat_epsilon_ = (self.epsilon - self.leaf_epsilon_) / (
                    self.max_depth
                )
            else:
                self.quantile_epsilon_ = 0
                self.node_num_epsilon_ = (self.epsilon - self.leaf_epsilon_) / (
                    self.max_depth
                )
                self.node_cat_epsilon_ = (self.epsilon - self.leaf_epsilon_) / (
                    self.max_depth
                )
        else:
            assert np.isclose(np.sum(self.epsilon_shares), 1), np.sum(
                self.epsilon_shares
            )
            self.quantile_epsilon_ = self.epsilon * self.epsilon_shares[0]
            self.node_num_epsilon_ = (
                self.epsilon * self.epsilon_shares[1] / self.max_depth
            )
            self.node_cat_epsilon_ = (
                self.epsilon
                * (self.epsilon_shares[0] + self.epsilon_shares[1])
                / self.max_depth
            )
            self.leaf_epsilon_ = self.epsilon * self.epsilon_shares[2]

            if not self.use_private_quantiles and self.quantile_epsilon_ != 0:
                warnings.warn(
                    "Spending privacy budget on private quantiles but use_private_quantiles is set to False"
                )

        assert np.isclose(
            self.node_num_epsilon_ * self.max_depth
            + self.leaf_epsilon_
            + self.quantile_epsilon_,
            self.epsilon,
        )
        assert np.isclose(
            self.node_cat_epsilon_ * self.max_depth + self.leaf_epsilon_,
            self.epsilon,
        )

        if self.verbose:
            print("epsilon distribution:")
            print("quantiles:", self.quantile_epsilon_)
            print("nodes (numerical):", self.node_num_epsilon_)
            print("nodes (categorical):", self.node_cat_epsilon_)
            print("leaves:", self.leaf_epsilon_)

    def __fit_recursive(self, X_binned, y, depth=0):
        if (
            depth == self.max_depth
            or len(np.unique(y)) == 1
            or len(X_binned) < self.min_samples_split
        ):
            return self.__create_leaf(y)

        feature, bin_order, split_bin, score = self.__find_best_split(
            X_binned, y, depth
        )

        # If no split improves the score then we stop and create a leaf
        if split_bin is None:
            if self.verbose:
                print("No threshold found!")
            return self.__create_leaf(y)

        mask_left = np.isin(X_binned[:, feature], bin_order[: split_bin + 1])
        mask_right = np.invert(mask_left)

        X_left = X_binned[mask_left]
        y_left = y[mask_left]
        X_right = X_binned[mask_right]
        y_right = y[mask_right]

        left_node = self.__fit_recursive(X_left, y_left, depth + 1)
        right_node = self.__fit_recursive(X_right, y_right, depth + 1)

        node = self.binner_.bin_to_node(feature, split_bin, bin_order)
        node.left_child = left_node
        node.right_child = right_node

        return node

    def __create_leaf(self, y):
        counts = list(np.bincount(y, minlength=len(self.classes_)))
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

        value = np.zeros(len(self.classes_))
        value[chosen_label] = 1
        return Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)

    def __find_best_split(self, X, y, depth):
        X_by_class = {}
        for label in self.classes_:
            X_by_class[label] = X[y == label]

        best_gini = float("inf")
        best_feature = None
        best_bin = None
        for feature_i in range(X.shape[1]):
            n_bins = self.n_bins_[feature_i]

            if self.binner_.categorical_features_[feature_i]:
                node_epsilon = self.node_cat_epsilon_
            else:
                node_epsilon = self.node_num_epsilon_

            # TODO: change from dict to numpy array

            histograms = {}
            for label in self.classes_:
                histograms[label] = private_bincount(
                    X_by_class[label][:, feature_i],
                    n_bins=n_bins,
                    epsilon=node_epsilon,
                    random_state=self.random_state_,
                )

            if self.binner_.categorical_features_[feature_i]:
                # NOTE: this method is only supported for binary classification.
                # Currently we do not handle multiclass categorical settings
                if len(self.classes_) != 2:
                    raise NotImplementedError(
                        "Multiclass categorical datasets are not yet supported"
                    )

                hist_total = np.array(
                    [histograms[label] for label in self.classes_]
                ).sum(axis=0)
                class_1_p = np.ones(n_bins) * 0.5
                np.divide(
                    histograms[self.classes_[1]],
                    hist_total,
                    out=class_1_p,
                    where=(hist_total != 0),
                )
                bin_order = np.argsort(class_1_p)
            else:
                bin_order = np.arange(self.n_bins_[feature_i])

            for label in self.classes_:
                histograms[label] = histograms[label][bin_order]

            left_counts = np.zeros(len(self.classes_), dtype=int)
            right_counts = np.zeros(len(self.classes_), dtype=int)
            for i in range(len(self.classes_)):
                right_counts[i] = histograms[self.classes_[i]].sum()

            for bin_i in range(len(bin_order)):
                for class_i, label in enumerate(self.classes_):
                    left_counts[class_i] += histograms[label][bin_i]
                    right_counts[class_i] -= histograms[label][bin_i]

                total_left = left_counts.sum()
                total_right = right_counts.sum()

                # Do not consider a split here if that would create a small leaf
                if (
                    total_left < self.min_samples_leaf
                    or total_right < self.min_samples_leaf
                ):
                    continue

                if total_left == 0:
                    gini_l = 0
                else:
                    denom = total_left**2
                    gini_l = 1 - np.square(left_counts).sum() / denom

                if total_right == 0:
                    gini_r = 0
                else:
                    denom = total_right**2
                    gini_r = 1 - np.square(right_counts).sum() / denom

                gini = ((total_left) * gini_l + (total_right) * gini_r) / (
                    total_left + total_right
                )

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_i
                    best_bin = bin_i
                    best_bin_order = bin_order

        return best_feature, best_bin_order, best_bin, best_gini

    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, "root_")

        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        return np.array(predictions)

    def predict(self, X):
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=1))
