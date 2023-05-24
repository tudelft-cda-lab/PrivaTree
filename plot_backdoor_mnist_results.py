import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")

results_df = pd.read_csv("out/backdoor_mnist_results.csv")

results_df["poisoned samples"] = results_df["poison rate"].apply(
    lambda r: int(r * 0.8 * 14000)
)

results_df["method"] = results_df["method"].replace(
    "PrivaTree (0.1)", "PrivaTree (ε = 0.1)"
)
results_df["method"] = results_df["method"].replace(
    "PrivaTree (0.01)", "PrivaTree (ε = 0.01)"
)

n_poison_samples = np.arange(results_df["poisoned samples"].max())
base_asr_01 = results_df[results_df["method"] == "PrivaTree (0.1)"][
    results_df["poison rate"] == 0
]["ASR"].mean()
base_asr_001 = results_df[results_df["method"] == "PrivaTree (0.01)"][
    results_df["poison rate"] == 0
]["ASR"].mean()
bound_01 = 1 - (1 - base_asr_01) * np.exp(-0.1 * n_poison_samples)
bound_001 = 1 - (1 - base_asr_001) * np.exp(-0.01 * n_poison_samples)

plt.subplots(figsize=(6.4, 3.0))
plt.plot(
    n_poison_samples,
    bound_01,
    c=sns.color_palette()[1],
    linestyle="--",
    label="bound (ε = 0.1)",
)
plt.plot(
    n_poison_samples,
    bound_001,
    c=sns.color_palette()[2],
    linestyle="--",
    label="bound (ε = 0.01)",
)
sns.lineplot(x="poisoned samples", y="ASR", hue="method", marker="o", data=results_df)

# Reorder legend items for nicer fit
handles, labels = plt.gca().get_legend_handles_labels()
order = [3, 0, 4, 1, 2]
plt.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="lower center",
    bbox_to_anchor=(0.48, 1.0),
    ncols=3,
)

plt.tight_layout()
plt.savefig("out/mnist_backdoor_asr_bounds.png", bbox_inches="tight")
plt.savefig("out/mnist_backdoor_asr_bounds.pdf", bbox_inches="tight")
plt.close()
