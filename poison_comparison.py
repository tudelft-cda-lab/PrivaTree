import sys
import time

import numpy as np
import openml
import pandas as pd
from diffprivlib.models import (
    DecisionTreeClassifier as DiffprivLibDecisionTreeClassifier,
    LogisticRegression as DiffPrivLibLogisticRegression,
)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from privatree.bdpt import BDPTClassifier
from privatree.dpa import DPAClassifier
from privatree.dpgdf import DPGDTClassifier
from privatree.privatree import PrivaTreeClassifier

def dpa_poison_accuracy_guarantee(poisoning_curve, data_size):
    n_poison_01 = int(0.001 * data_size)
    n_poison_05 = int(0.005 * data_size)
    n_poison_1 = int(0.01 * data_size)

    if n_poison_01 >= len(poisoning_curve):
        guarantee_01 = 0
    else:
        guarantee_01 = poisoning_curve[n_poison_01]

    if n_poison_05 >= len(poisoning_curve):
        guarantee_05 = 0
    else:
        guarantee_05 = poisoning_curve[n_poison_05]

    if n_poison_1 >= len(poisoning_curve):
        guarantee_1 = 0
    else:
        guarantee_1 = poisoning_curve[n_poison_1]

    return guarantee_01, guarantee_05, guarantee_1

def epsilon_poison_accuracy_guarantee(base_accuracy, epsilon, data_size):
    """Compute a differential privacy guarantee on poison accuracy under 0.5%, 1% and 2% of poison samples."""
    n_poison_01 = int(0.001 * data_size)
    n_poison_05 = int(0.005 * data_size)
    n_poison_1 = int(0.01 * data_size)
    return (
        base_accuracy * np.exp(-epsilon * n_poison_01),
        base_accuracy * np.exp(-epsilon * n_poison_05),
        base_accuracy * np.exp(-epsilon * n_poison_1),
    )

max_bins = 10
max_depths = [4]
epsilons = [0.001, 0.01, 0.1, 1.0]
n_splits = 5

assert len(sys.argv) == 2

benchmark = sys.argv[1]

output_filename_prefix = "out/benchmark_poisoning"

if benchmark == "categorical":
    SUITE_ID = 334  # Classification on numerical and categorical features
    output_filename = output_filename_prefix + "_categorical.csv"

    benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

    task_ids = benchmark_suite.tasks
elif benchmark == "numerical":
    SUITE_ID = 337  # Classification on numerical features
    output_filename = output_filename_prefix + "_numerical.csv"

    benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

    task_ids = benchmark_suite.tasks
elif benchmark == "uci":
    task_ids = [
        15,  # breast cancer (Wisconsin)
        24,  # mushroom
        37,  # diabetes
        56,  # vote
        959,  # nursery
        1590,  # adult
    ]
    output_filename = output_filename_prefix + "_uci.csv"
else:
    raise ValueError(f"Unknown benchmark {benchmark}")

random_state = check_random_state(1)

results = []
for task_id in task_ids:  # iterate over all tasks
    if benchmark == "uci":
        dataset = openml.datasets.get_dataset(
            task_id
        )  # download the OpenML dataset directly
    else:
        task = openml.tasks.get_task(task_id)  # download the OpenML task

        dataset = task.get_dataset()

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )

    # Drop rows with NaNs
    keep_rows = ~np.any(np.isnan(X), axis=1)
    X = X[keep_rows]
    y = y[keep_rows]

    print(f"Starting {dataset.name} ({X.shape})")

    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_i, (train_indices, test_indices) in enumerate(k_fold.split(X, y)):
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        # Scale the data for logistic regression
        scaler = StandardScaler()
        X_train_lr = scaler.fit_transform(X_train)
        X_test_lr = scaler.transform(X_test)

        # Compute feature ranges on the train data (these are public information)
        feature_range = np.concatenate(
            (X_train.min(axis=0).reshape(-1, 1), X_train.max(axis=0).reshape(-1, 1)),
            axis=1,
        )
        bounds = (feature_range[:, 0], feature_range[:, 1])
        data_norm = np.linalg.norm(X_train_lr, ord=2, axis=1).max()

        # Compute number of categories if they exist (public information)
        categorical_features = []
        for feature_i, is_categorical in enumerate(categorical_indicator):
            if not is_categorical:
                categorical_features.append(0)
            else:
                categorical_features.append(int(X_train[:, feature_i].max() + 1))
        categorical_features = np.array(categorical_features)

        for epsilon in epsilons:
            start_time = time.time()
            logistic_regression = DiffPrivLibLogisticRegression(
                random_state=random_state,
                epsilon=epsilon,
                data_norm=data_norm,
            )
            # logistic_regression.fit(X_train, y_train)
            logistic_regression.fit(X_train_lr, y_train)
            runtime = time.time() - start_time
            # train_accuracy = logistic_regression.score(X_train, y_train)
            # test_accuracy = logistic_regression.score(X_test, y_test)
            train_accuracy = logistic_regression.score(X_train_lr, y_train)
            test_accuracy = logistic_regression.score(X_test_lr, y_test)
            results.append(
                (
                    dataset.name,
                    fold_i,
                    "logistic regression",
                    None,
                    epsilon,
                    None,
                    train_accuracy,
                    test_accuracy,
                    runtime,
                    None,
                    *epsilon_poison_accuracy_guarantee(test_accuracy, epsilon, len(X_train)),
                )
            )

            print(results[-1])

        for max_depth in max_depths:
            start_time = time.time()
            tree = DecisionTreeClassifier(
                max_depth=max_depth, random_state=random_state
            )
            tree.fit(X_train, y_train)
            runtime = time.time() - start_time
            train_accuracy = tree.score(X_train, y_train)
            test_accuracy = tree.score(X_test, y_test)
            results.append(
                (
                    dataset.name,
                    fold_i,
                    "regular tree",
                    max_depth,
                    None,
                    None,
                    train_accuracy,
                    test_accuracy,
                    runtime,
                    None,
                    None,
                    None,
                    None,
                )
            )

            print(results[-1])

            for n_partitions in [5, 10, 50, 100, 500, 1000, 5000]:
                # Skip training if there are too many partitions for the data size
                if n_partitions > 0.5 * len(X_train):
                    continue

                start_time = time.time()
                dpa = DPAClassifier(
                    n_partitions=n_partitions, max_depth=max_depth, random_state=random_state
                )
                dpa.fit(X_train, y_train)
                runtime = time.time() - start_time
                train_accuracy = dpa.score(X_train, y_train)
                test_accuracy = dpa.score(X_test, y_test)
                poisoning_curve = dpa.poisoning_accuracy_curve(X_test, y_test)
                results.append(
                    (
                        dataset.name,
                        fold_i,
                        "DPA",
                        max_depth,
                        None,
                        None,
                        train_accuracy,
                        test_accuracy,
                        runtime,
                        n_partitions,
                        *dpa_poison_accuracy_guarantee(poisoning_curve, len(X_train)),
                    )
                )

                print(results[-1])

            for epsilon in epsilons:
                start_time = time.time()
                diffprivlib_tree = DiffprivLibDecisionTreeClassifier(
                    max_depth=max_depth,
                    random_state=random_state,
                    epsilon=epsilon,
                    bounds=bounds,
                    classes=[0, 1],
                )
                diffprivlib_tree.fit(X_train, y_train)
                runtime = time.time() - start_time
                train_accuracy = diffprivlib_tree.score(X_train, y_train)
                test_accuracy = diffprivlib_tree.score(X_test, y_test)
                results.append(
                    (
                        dataset.name,
                        fold_i,
                        "diffprivlib tree",
                        max_depth,
                        epsilon,
                        None,
                        train_accuracy,
                        test_accuracy,
                        runtime,
                        None,
                        *epsilon_poison_accuracy_guarantee(test_accuracy, epsilon, len(X_train)),
                    )
                )

                print(results[-1])

                start_time = time.time()
                private_tree = PrivaTreeClassifier(
                    max_depth=max_depth,
                    max_bins=max_bins,
                    epsilon=epsilon,
                    feature_range=feature_range,
                    categorical_features=categorical_features,
                    random_state=random_state,
                )
                private_tree.fit(X_train, y_train)
                runtime = time.time() - start_time
                train_accuracy = private_tree.score(X_train, y_train)
                test_accuracy = private_tree.score(X_test, y_test)
                results.append(
                    (
                        dataset.name,
                        fold_i,
                        "PrivaTree",
                        max_depth,
                        epsilon,
                        max_bins,
                        train_accuracy,
                        test_accuracy,
                        runtime,
                        None,
                        *epsilon_poison_accuracy_guarantee(test_accuracy, epsilon, len(X_train)),
                    )
                )

                print(results[-1])

                results_df = pd.DataFrame(
                    results,
                    columns=[
                        "dataset",
                        "fold",
                        "method",
                        "max_depth",
                        "epsilon",
                        "max_bins",
                        "train accuracy",
                        "test accuracy",
                        "runtime",
                        "n_partitions",
                        "0.1% guarantee",
                        "0.5% guarantee",
                        "1% guarantee",
                    ],
                )
                results_df.to_csv(output_filename, index=False)
