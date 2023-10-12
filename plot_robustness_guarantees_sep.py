import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from diffprivlib.models import DecisionTreeClassifier as DiffPrivLibTree
from diffprivlib.models import LogisticRegression as DiffPrivLibLogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from tqdm import tqdm

from privatree.dpa import DPAClassifier
from privatree.privatree import PrivaTreeClassifier


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

scores = []
random_state = check_random_state(0)

# TODO: specify dataset properties to remove warnings

splitter = ShuffleSplit(
    n_splits=n_splits, test_size=test_size, random_state=random_state
)
dpa_curves = []
for train_i, test_i in tqdm(splitter.split(X, y), total=n_splits):
    X_train = X[train_i]
    y_train = y[train_i]
    X_test = X[test_i]
    y_test = y[test_i]

    # Separate categorical features for DPGDF
    X_train_cat = X_train[:, categorical_indicator]
    X_test_cat = X_test[:, categorical_indicator]

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X_train, y_train)
    accuracy = tree.score(X_test, y_test)
    scores.append((accuracy, "Decision tree", None))

    tree = DiffPrivLibTree(max_depth=max_depth, epsilon=0.1, random_state=random_state)
    tree.fit(X_train, y_train)
    accuracy = tree.score(X_test, y_test)
    scores.append((accuracy, "DiffPrivLib tree (0.1)", 0.1))

    tree = PrivaTreeClassifier(
        max_depth=max_depth,
        epsilon=0.1,
        categorical_features=categorical_indicator,
        random_state=random_state,
    )
    tree.fit(X_train, y_train)
    accuracy = tree.score(X_test, y_test)
    scores.append((accuracy, "PrivaTree (0.1)", 0.1))

    tree = DiffPrivLibTree(max_depth=max_depth, epsilon=0.01, random_state=random_state)
    tree.fit(X_train, y_train)
    accuracy = tree.score(X_test, y_test)
    scores.append((accuracy, "DiffPrivLib tree (0.01)", 0.01))

    tree = PrivaTreeClassifier(
        max_depth=max_depth,
        epsilon=0.01,
        categorical_features=categorical_indicator,
        random_state=random_state,
    )
    tree.fit(X_train, y_train)
    accuracy = tree.score(X_test, y_test)
    scores.append((accuracy, "PrivaTree (0.01)", 0.01))

    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    X_test_lr = scaler.transform(X_test)
    data_norm = np.linalg.norm(X_train_lr, ord=2, axis=1).max()

    logistic_regression = DiffPrivLibLogisticRegression(
        random_state=random_state,
        epsilon=0.1,
        data_norm=data_norm,
    )
    logistic_regression.fit(X_train_lr, y_train)
    accuracy = logistic_regression.score(X_test_lr, y_test)
    scores.append((accuracy, "DiffPrivLib LR (0.1)", 0.1))

    logistic_regression = DiffPrivLibLogisticRegression(
        random_state=random_state,
        epsilon=0.01,
        data_norm=data_norm,
    )
    logistic_regression.fit(X_train_lr, y_train)
    accuracy = logistic_regression.score(X_test_lr, y_test)
    scores.append((accuracy, "DiffPrivLib LR (0.01)", 0.01))

    dpa = DPAClassifier(n_partitions=1000, max_depth=4, random_state=1)
    dpa.fit(X_train, y_train)
    accuracy = dpa.score(X_test, y_test)
    scores.append((accuracy, "DPA (1000 trees)", None))
    dpa_curves.append(dpa.poisoning_accuracy_curve(X_test, y_test))

dpa_curves = np.array(dpa_curves)

scores_df = pd.DataFrame(scores, columns=["test accuracy", "method", "epsilon"])
scores_dict = scores_df.groupby("method")["test accuracy"].mean().to_dict()


def plot(max_poison_samples, epsilon, ax):
    n_poison_samples = np.arange(max_poison_samples + 1)

    if epsilon == 0.1:
        dt_label = "decision tree"
        dpa_label = "DPA (1000 trees)"
    else:
        dt_label = None
        dpa_label = None

    regular_tree_score = scores_dict["Decision tree"]
    sns.scatterplot(
        x=[0],
        y=[regular_tree_score],
        color=sns.color_palette()[0],
        label=dt_label,
        ax=ax,
    )

    if epsilon == 0.1:
        linestyle = "solid"
    elif epsilon == 0.01:
        linestyle = "dashed"
    else:
        raise ValueError()

    ax.axvline(x=0, color="gray", zorder=-1)

    guarantees = (
        np.exp(-n_poison_samples * epsilon)
        * scores_dict[f"DiffPrivLib tree ({epsilon})"]
    )
    sns.lineplot(
        x=n_poison_samples,
        y=guarantees,
        color=sns.color_palette()[1],
        linewidth=3,
        linestyle=linestyle,
        label=f"DiffPrivLib tree ($\epsilon$ = {epsilon})",
        ax=ax,
    )

    guarantees = (
        np.exp(-n_poison_samples * epsilon) * scores_dict[f"PrivaTree ({epsilon})"]
    )
    sns.lineplot(
        x=n_poison_samples,
        y=guarantees,
        color=sns.color_palette()[2],
        linewidth=2,
        linestyle=linestyle,
        label=f"PrivaTree ($\epsilon$ = {epsilon})",
        ax=ax,
    )

    guarantees = (
        np.exp(-n_poison_samples * epsilon) * scores_dict[f"DiffPrivLib LR ({epsilon})"]
    )
    sns.lineplot(
        x=n_poison_samples,
        y=guarantees,
        color=sns.color_palette()[4],
        linewidth=1.5,
        linestyle=linestyle,
        label=f"DiffPrivLib LR ($\epsilon$ = {epsilon})",
        ax=ax,
    )

    length = min(len(n_poison_samples), dpa_curves.shape[1])
    guarantees = np.mean(dpa_curves, axis=0)
    sns.lineplot(
        x=n_poison_samples[:length],
        y=guarantees[:length],
        color="gray",
        linestyle="dotted",
        linewidth=1.5,
        label=dpa_label,
        ax=ax,
    )

    ax.get_legend().remove()

    if max_poison_samples > 10:
        ax.set_xlim(-max_poison_samples // 1000, max_poison_samples)

        # max_poison_samples is 1% of data so we can use it to plot the other percentages
        # on top of the graph axis

        extra_axis = matplotlib.axis.XAxis(ax)
        extra_axis.tick_top()
        extra_axis.set_ticks(
            [max_poison_samples * mul for mul in (0, 0.2, 0.4, 0.6, 0.8, 1)]
        )
        extra_axis.set_ticklabels(["0%", "0.2%", "0.4%", "0.6%", "0.8%", "1%"])
        extra_axis.set_label_position("top")
        ax.add_artist(extra_axis)
    else:
        ax.set_xlim(-0.05, max_poison_samples)

        assert max_poison_samples == 4
        ax.set_xticks([0, 2, 4])


_, ax = plt.subplots(2, 2, figsize=(6.5, 4.5), gridspec_kw={"width_ratios": [1, 3]})

one_percent_data_size = len(X) * (1 - test_size) // 100
plot(4, epsilon=0.1, ax=ax[0, 0])
plot(one_percent_data_size, epsilon=0.1, ax=ax[0, 1])
plot(4, epsilon=0.01, ax=ax[1, 0])
plot(one_percent_data_size, epsilon=0.01, ax=ax[1, 1])
ax[0, 0].set_ylabel("accuracy guarantee")
ax[1, 0].set_xlabel("poisoned samples")
ax[1, 1].set_xlabel("poisoned samples")

ax[0, 1].legend()
ax[1, 1].legend()

plt.tight_layout()
plt.savefig(f"out/robustness_guarantees_sep_{data_id}.png", bbox_inches="tight")
plt.savefig(f"out/robustness_guarantees_sep_{data_id}.pdf", bbox_inches="tight")
plt.close()
