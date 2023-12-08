import sys

import matplotlib.pyplot as plt
import numpy as np
import openml
import seaborn as sns
import ternary

from privatree.privatree import PrivaTreeClassifier

sns.set_theme(style="white", palette="colorblind")

epsilon = 0.1

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


def tree_score(epsilon_shares, repetitions=50):
    if any(epsilon == 0 for epsilon in epsilon_shares):
        return 0.0

    total = 0
    for random_state in range(repetitions):
        tree = PrivaTreeClassifier(
            max_depth=3,
            epsilon=epsilon,
            max_bins=10,
            categorical_features=categorical_indicator,
            epsilon_shares=epsilon_shares,
            use_uniform_bins=False,
            random_state=random_state,
        )
        tree.fit(X, y)
        total += tree.score(X, y)

    print(".", end="", flush=True)
    return total / repetitions


scale = 20

_, tax = ternary.figure(scale=scale)
tax.heatmapf(tree_score, boundary=False, style="hexagonal")
tax.boundary(linewidth=2.0)
tax.gridlines(color="black", multiple=1)
tax.left_axis_label("$\\epsilon_{leaves}$", offset=0.18)
tax.right_axis_label("$\\epsilon_{nodes}$", offset=0.18)
tax.bottom_axis_label("$\\epsilon_{quantiles}$", offset=0.02)

# Set and format axes ticks.
ticks = [
    f"{epsilon * (i / float(scale)):.2f}".replace("0.", ".") for i in range(scale + 1)
]
tax.ticks(ticks=ticks, axis="lr", linewidth=1, offset=0.03, tick_formats="%0.2f")
tax.ticks(ticks=ticks, axis="b", linewidth=1, offset=0.02, tick_formats="%0.2f")

tax.get_axes().axis("off")
tax.clear_matplotlib_ticks()
tax._redraw_labels()
plt.tight_layout()
tax.savefig(f"out/ternary_{data_id}.png", bbox_inches="tight")
tax.savefig(f"out/ternary_{data_id}.pdf", bbox_inches="tight")
tax.close()
