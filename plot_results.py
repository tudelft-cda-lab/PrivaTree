import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")

for result_type in ["numerical", "categorical", "uci"]:
    result_filename = f"out/benchmark_results_{result_type}.csv"

    if not os.path.isfile(result_filename):
        continue

    print(result_type)

    results_df = pd.read_csv(result_filename)

    depth_3_results = results_df[results_df["max_depth"] == 3]

    hue_order = [
        "diffprivlib tree",
        "DPGDT",
        "BDPT",
        "private tree",
        "private tree (non-priv. quantiles)",
    ]

    for epsilon in [0.01, 0.1, 1.0]:
        print(epsilon)

        eps_results = results_df[
            (results_df["epsilon"] == epsilon) | (results_df["epsilon"].isna())
        ]

        epsilon_text = str(epsilon).replace(".", "_")

        sns.barplot(
            data=eps_results,
            x="dataset",
            y="test accuracy",
            hue="method",
            hue_order=hue_order,
        )
        plt.tight_layout()
        plt.savefig(f"out/test_accuracy_datasets_{result_type}_{epsilon_text}.png")
        plt.savefig(f"out/test_accuracy_datasets_{result_type}_{epsilon_text}.pdf")
        plt.close()

        # This reorders the columns and removes the 'dummy' column
        column_order = [
            "dataset",
            "regular tree",
            "BDPT",
            "private tree (non-priv. quantiles)",
            "DPGDT",
            "diffprivlib tree",
            "private tree",
        ]

        if result_type == "numerical":
            eps_results.loc[eps_results["method"] == "DPGDT", "test accuracy"] = -1
            eps_results.loc[eps_results["method"] == "DPGDT", "runtime"] = -1

        mean_table = pd.pivot_table(
            eps_results,
            index="dataset",
            columns="method",
            values="test accuracy",
            aggfunc="mean",
        ).reset_index()
        mean_table = mean_table[column_order]

        sem_table = pd.pivot_table(
            eps_results,
            index="dataset",
            columns="method",
            values="test accuracy",
            aggfunc="sem",
        ).reset_index()
        sem_table = sem_table[column_order]

        def concat_and_format(mean, sem):
            if isinstance(mean, str):
                return mean

            return f"{mean:.3f} \\tiny $\\pm$ {sem:.3f}"

        concat_and_format_vec = np.vectorize(concat_and_format)
        formatted_scores = pd.DataFrame(
            concat_and_format_vec(mean_table, sem_table), columns=column_order
        )

        method_mapping = {
            "regular tree": "decision tree",
            "diffprivlib tree": "diffprivlib",
            "private tree": "PrivaTree",
            "private tree (non-priv. quantiles)": "PrivaTree leaking splits",
        }
        formatted_scores = formatted_scores.rename(columns=method_mapping)
        formatted_scores["dataset"] = formatted_scores["dataset"].apply(
            lambda x: x.replace("_", "\\_")
        )
        print(formatted_scores.to_csv(index=False))

        if epsilon == 0.1:
            print("runtime data:")
            mean_runtime_table = pd.pivot_table(
                eps_results,
                index="dataset",
                columns="method",
                values="runtime",
                aggfunc="mean",
            ).reset_index()
            mean_runtime_table = mean_runtime_table[column_order]

            sem_runtime_table = pd.pivot_table(
                eps_results,
                index="dataset",
                columns="method",
                values="runtime",
                aggfunc="sem",
            ).reset_index()
            sem_runtime_table = sem_runtime_table[column_order]

            def concat_and_format(mean, sem):
                if isinstance(mean, str):
                    return mean

                if np.isnan(mean):
                    return "-"

                if round(mean) == 0:
                    return f"<1 \\tiny $\\pm$ {sem:.0f}"

                return f"{mean:.0f} \\tiny $\\pm$ {sem:.0f}"

            concat_and_format_vec = np.vectorize(concat_and_format)
            formatted_scores = pd.DataFrame(
                concat_and_format_vec(mean_runtime_table, sem_runtime_table),
                columns=column_order,
            )

            method_mapping = {
                "regular tree": "decision tree",
                "diffprivlib tree": "diffprivlib",
                "private tree": "PrivaTree",
                "private tree (non-priv. quantiles)": "PrivaTree leaking splits",
            }
            formatted_scores = formatted_scores.rename(columns=method_mapping)
            formatted_scores["dataset"] = formatted_scores["dataset"].apply(
                lambda x: x.replace("_", "\\_")
            )
            print(formatted_scores.to_csv(index=False))
