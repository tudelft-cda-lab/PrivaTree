import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from tqdm import tqdm

from privatree.privatree import PrivaTreeClassifier


def get_badnets_patch():
    """
    In this attack we create a trigger patch for the MNIST dataset
    that exists of a pattern of 4 pixels in the bottom right.

    Returns a vector of trigger pixels (features)
    """
    reshaped = np.arange(28 * 28).reshape(28, 28)
    indices = [(27, 27), (25, 27), (27, 25), (26, 26)]
    return np.array([reshaped[i[0], i[1]] for i in indices])


def apply_trigger(X, trigger):
    X = X.copy()
    X[:, trigger] = 255
    return X


repetitions = 50
test_size = 0.2
max_depth = 4

poison_shares = np.linspace(0.0, 0.01, 11)

# Download the MNIST dataset
dataset = openml.datasets.get_dataset(554)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

# Filter for only the class 0 and class 1
X = X[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]

trigger = get_badnets_patch()
print("trigger pattern:", trigger)

data_random_state = check_random_state(0)
random_state = check_random_state(0)

results = []
for _ in tqdm(range(repetitions)):
    for poison_share in poison_shares:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state=data_random_state
        )

        n_poison_samples = int(poison_share * X_train.shape[0])

        if poison_share == 0:
            X_train_poisoned = X_train
            y_train_poisoned = y_train
        else:
            i_train_zeros = np.where(y_train == 0)[0]
            i_poison = data_random_state.choice(
                i_train_zeros, size=n_poison_samples, replace=False
            )
            X_poison = apply_trigger(X_train[i_poison], trigger)
            y_poison = np.ones(X_poison.shape[0], dtype=int)

            X_train_poisoned = np.concatenate((X_train, X_poison), axis=0)
            y_train_poisoned = np.concatenate((y_train, y_poison), axis=0)

            # Plot and save a poisoned image
            sns.set_theme(style="white")
            plt.imshow(X_poison[0].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
            plt.xticks(())
            plt.yticks(())
            plt.tight_layout()
            plt.savefig("out/mnist_backdoor_image.png", bbox_inches="tight")
            plt.savefig("out/mnist_backdoor_image.pdf", bbox_inches="tight")
            plt.close()

        # Create a test set with only 0 samples that have a patch and are
        # labeled as 1 to evaluate the attack success rate.
        X_test_asr = X_test[y_test == 0]
        X_test_asr = apply_trigger(X_test_asr, trigger)
        y_test_asr = np.ones(X_test_asr.shape[0], dtype=y_test.dtype)

        for method in ["decision tree", "PrivaTree (0.1)", "PrivaTree (0.01)"]:
            if method == "decision tree":
                tree = DecisionTreeClassifier(
                    max_depth=max_depth, random_state=random_state
                )
            elif method.lower().startswith("privatree"):
                epsilon = float(method.split("(")[1].split(")")[0])
                tree = PrivaTreeClassifier(
                    max_depth=max_depth, epsilon=epsilon, random_state=random_state
                )
            else:
                raise ValueError("Unknown method")

            tree.fit(X_train_poisoned, y_train_poisoned)
            accuracy = tree.score(X_test, y_test)

            attack_success_rate = tree.score(X_test_asr, y_test_asr)

            results.append(
                (poison_share, n_poison_samples, method, accuracy, attack_success_rate)
            )

    results_df = pd.DataFrame(
        results,
        columns=["poison rate", "poisoned samples", "method", "accuracy", "ASR"],
    )
    results_df.to_csv("out/backdoor_mnist_results.csv", index=False)

    sns.set_theme(style="whitegrid", palette="colorblind")
    sns.lineplot(x="poison rate", y="ASR", hue="method", marker="o", data=results_df)
    plt.tight_layout()
    plt.savefig("out/mnist_backdoor_asr.png", bbox_inches="tight")
    plt.savefig("out/mnist_backdoor_asr.pdf", bbox_inches="tight")
    plt.close()
