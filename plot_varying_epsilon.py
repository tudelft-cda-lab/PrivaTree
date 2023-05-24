import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from diffprivlib.models import DecisionTreeClassifier as DiffPrivLibTree
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from tqdm import tqdm

from privatree.bdpt import BDPTClassifier
from privatree.dpgdf import DPGDTClassifier
from privatree.privatree import PrivaTreeClassifier


def plot_figure(results_df, data_id, baseline):
    results_df["method"] = results_df["method"].replace("DPGDT", "DPGDF")

    # Make sure to not plot anything too far below the baseline (majority class prediction)
    # because it will skew the axes by a lot
    max_score = results_df.groupby("method").mean()["test accuracy"].max()
    min_score = baseline
    margin = 0.1 * (max_score - min_score)

    _, ax = plt.subplots(figsize=(5.5, 3.5))
    sns.lineplot(
        x="epsilon", y="test accuracy", hue="method", marker="o", data=results_df
    )
    ax.axhline(y=baseline, color="gray", ls="dashed", label="majority class")
    plt.legend(loc="lower center", bbox_to_anchor=(0.43, 1.0), ncols=3)
    plt.xscale("log")
    plt.xlabel("$\\epsilon$")
    plt.ylim(min_score - margin, max_score + margin)
    plt.tight_layout()
    plt.savefig(f"out/varying_epsilon_{data_id}.png", bbox_inches="tight")
    plt.savefig(f"out/varying_epsilon_{data_id}.pdf", bbox_inches="tight")
    plt.close()


max_depth = 4
test_size = 0.2
n_splits = 50

sns.set_theme(style="whitegrid", palette="colorblind")

assert len(sys.argv) == 2

data_id = int(sys.argv[1])

dataset = openml.datasets.get_dataset(data_id)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

# Drop rows with NaNs
keep_rows = ~np.any(np.isnan(X), axis=1)
X = X[keep_rows]
y = y[keep_rows]

counts = np.bincount(y, minlength=2)
baseline = max(counts[0] / counts.sum(), counts[1] / counts.sum())

if not os.path.isfile(f"out/varying_epsilon_{data_id}.csv"):
    results = []
    # for epsilon in np.logspace(-3, 1, 9, base=10):
    random_state = check_random_state(0)
    for epsilon in np.logspace(-3, 1, 5, base=10):
        splitter = ShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )

        for train_i, test_i in tqdm(splitter.split(X, y), total=n_splits):
            X_train = X[train_i]
            y_train = y[train_i]
            X_test = X[test_i]
            y_test = y[test_i]

            # Separate categorical features for DPGDF
            X_train_cat = X_train[:, categorical_indicator]
            X_test_cat = X_test[:, categorical_indicator]

            tree = DecisionTreeClassifier(
                max_depth=max_depth, random_state=random_state
            )
            tree.fit(X_train, y_train)
            accuracy = tree.score(X_test, y_test)
            results.append((epsilon, accuracy, "Decision tree"))

            tree = DiffPrivLibTree(
                max_depth=max_depth, epsilon=epsilon, random_state=random_state
            )
            tree.fit(X_train, y_train)
            accuracy = tree.score(X_test, y_test)
            results.append((epsilon, accuracy, "DiffPrivLib"))

            tree = PrivaTreeClassifier(
                max_depth=max_depth,
                epsilon=epsilon,
                categorical_features=categorical_indicator,
                random_state=random_state,
            )
            tree.fit(X_train, y_train)
            accuracy = tree.score(X_test, y_test)
            results.append((epsilon, accuracy, "PrivaTree (ours)"))

            tree = BDPTClassifier(
                max_depth=max_depth, epsilon=epsilon, random_state=random_state
            )
            tree.fit(X_train, y_train)
            accuracy = tree.score(X_test, y_test)
            results.append((epsilon, accuracy, "BDPT"))

            if X_train_cat.shape[1] > 0:
                tree = DPGDTClassifier(
                    max_depth=max_depth,
                    epsilon=epsilon,
                    categorical_features=True,
                    random_state=random_state,
                )
                tree.fit(X_train_cat, y_train)
                accuracy = tree.score(X_test_cat, y_test)
                results.append((epsilon, accuracy, "DPGDT"))

        results_df = pd.DataFrame(
            results, columns=["epsilon", "test accuracy", "method"]
        )
        results_df.to_csv(f"out/varying_epsilon_{data_id}.csv")
        plot_figure(results_df, data_id, baseline)
else:
    results_df = pd.read_csv(f"out/varying_epsilon_{data_id}.csv")
    plot_figure(results_df, data_id, baseline)
