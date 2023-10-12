import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="colorblind")

# filename = "out/benchmark_poisoning_numerical.csv"
# filename = "out/benchmark_poisoning_categorical.csv"
filename = "out/benchmark_poisoning_uci.csv"

results_df = pd.read_csv(filename)
results_df = results_df[
    (results_df["method"].isin(["PrivaTree", "logistic regression"]))
    & (results_df["epsilon"].isin([0.01, 0.1]))
]

# Average over the folds
results_df = (
    results_df.groupby(["dataset", "method", "epsilon"])[
        ["test accuracy", "0.1% guarantee", "0.5% guarantee", "1% guarantee"]
    ]
    .mean()
    .reset_index()
)

print(
    results_df.groupby(["dataset", "method", "epsilon"])[
        ["test accuracy", "0.1% guarantee", "0.5% guarantee", "1% guarantee"]
    ]
    .mean()
    .to_csv(float_format=lambda x: round(x, 3))
)
