import sys
import time

import numpy as np
import openml
import pandas as pd
from diffprivlib.models import (
    DecisionTreeClassifier as DiffprivLibDecisionTreeClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from privatree.bdpt import BDPTClassifier
from privatree.dpgdf import DPGDTClassifier
from privatree.privatree import PrivaTreeClassifier

max_bins = 10
max_depths = [4]
epsilons = [0.01, 0.1, 1.0]  # [0.01, 0.1, 1.0, 10.0]
n_splits = 5

assert len(sys.argv) == 2

benchmark = sys.argv[1]

if benchmark == "categorical":
    SUITE_ID = 334  # Classification on numerical and categorical features
    output_filename = "out/benchmark_results_categorical.csv"

    benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

    task_ids = benchmark_suite.tasks
elif benchmark == "numerical":
    SUITE_ID = 337  # Classification on numerical features
    output_filename = "out/benchmark_results_numerical.csv"

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
    output_filename = "out/benchmark_results_uci.csv"
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

        # Separate categorical features for DPGDF
        X_train_cat = X_train[:, categorical_indicator]
        X_test_cat = X_test[:, categorical_indicator]

        # Compute feature ranges on the train data (these are public information)
        feature_range = np.concatenate(
            (X_train.min(axis=0).reshape(-1, 1), X_train.max(axis=0).reshape(-1, 1)),
            axis=1,
        )
        bounds = (feature_range[:, 0], feature_range[:, 1])

        # Compute number of categories if they exist (public information)
        categorical_features = []
        for feature_i, is_categorical in enumerate(categorical_indicator):
            if not is_categorical:
                categorical_features.append(0)
            else:
                categorical_features.append(int(X_train[:, feature_i].max() + 1))
        categorical_features = np.array(categorical_features)

        # Train a dummy classifier that predicts the majority class as baseline
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dummy_train_accuracy = dummy.score(X_train, y_train)
        dummy_test_accuracy = dummy.score(X_test, y_test)
        print("Dummy:", dummy_test_accuracy)

        results.append(
            (
                dataset.name,
                fold_i,
                "dummy",
                0,
                None,
                None,
                dummy_train_accuracy,
                dummy_test_accuracy,
                None,
            )
        )

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
                        "private tree",
                        max_depth,
                        epsilon,
                        max_bins,
                        train_accuracy,
                        test_accuracy,
                        runtime,
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
                    use_private_quantiles=False,
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
                        "private tree (non-priv. quantiles)",
                        max_depth,
                        epsilon,
                        max_bins,
                        train_accuracy,
                        test_accuracy,
                        runtime,
                    )
                )

                print(results[-1])

                # For DPGDF we train on categorical features only as the algorithm
                # does not support numerical features (same as Borhan, 2018)
                if X_train_cat.shape[1] == 0:
                    results.append(
                        (
                            dataset.name,
                            fold_i,
                            "DPGDT",
                            max_depth,
                            epsilon,
                            None,
                            None,
                            None,
                            None,
                        )
                    )
                else:
                    start_time = time.time()
                    dpgd_tree = DPGDTClassifier(
                        max_depth=max_depth,
                        epsilon=epsilon,
                        feature_range=feature_range,
                        categorical_features=True,
                        random_state=random_state,
                    )
                    dpgd_tree.fit(X_train_cat, y_train)
                    runtime = time.time() - start_time
                    train_accuracy = dpgd_tree.score(X_train_cat, y_train)
                    test_accuracy = dpgd_tree.score(X_test_cat, y_test)
                    results.append(
                        (
                            dataset.name,
                            fold_i,
                            "DPGDT",
                            max_depth,
                            epsilon,
                            None,
                            train_accuracy,
                            test_accuracy,
                            runtime,
                        )
                    )

                print(results[-1])

                if dataset.name.lower() == "higgs":
                    # This algorithm takes too long to run on higgs (> 2 hours)
                    results.append(
                        (
                            dataset.name,
                            fold_i,
                            "BDPT",
                            max_depth,
                            epsilon,
                            None,
                            None,
                            None,
                            None,
                        )
                    )
                else:
                    start_time = time.time()
                    bdpt_tree = BDPTClassifier(
                        max_depth=max_depth,
                        epsilon=epsilon,
                        feature_range=feature_range,
                        categorical_features=categorical_features,
                        random_state=random_state,
                    )
                    bdpt_tree.fit(X_train, y_train)
                    runtime = time.time() - start_time
                    train_accuracy = bdpt_tree.score(X_train, y_train)
                    test_accuracy = bdpt_tree.score(X_test, y_test)
                    results.append(
                        (
                            dataset.name,
                            fold_i,
                            "BDPT",
                            max_depth,
                            epsilon,
                            None,
                            train_accuracy,
                            test_accuracy,
                            runtime,
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
                    ],
                )
                results_df.to_csv(output_filename, index=False)
