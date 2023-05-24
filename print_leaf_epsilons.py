import numpy as np
from scipy.optimize import minimize_scalar

from privatree.privatree import _worst_expected_error_pf


def _worst_expected_error_em(n_classes):
    def neg_f(p):
        # This is the negative of the expression that describes the worst-case
        # expected error of permute-and-flip for n candidates.
        # Here delta=1 because adding 1 item to the database increases
        # sample counts by 1.
        return -2 * np.log(1 / p) * (1 - (1 / (1 + (n_classes - 1) * p)))

    # we take the negative because we want to maximize
    return -minimize_scalar(neg_f, bounds=(0, 1)).fun


max_depth = 4
n_classes = 2

print("Permute-and-flip")
for n_samples in [1_000, 10_000, 100_000]:
    for leaf_label_success_prob in [0.9, 0.95, 0.99]:
        required_leaf_epsilon = (
            _worst_expected_error_pf(n_classes) * (2**max_depth)
        ) / (n_samples * (1 - leaf_label_success_prob))

        print(f"{n_samples}, {leaf_label_success_prob}: {required_leaf_epsilon:.3f}")

print("Exponential mechanism")
for n_samples in [1_000, 10_000, 100_000]:
    for leaf_label_success_prob in [0.9, 0.95, 0.99]:
        required_leaf_epsilon = (
            _worst_expected_error_em(n_classes) * (2**max_depth)
        ) / (n_samples * (1 - leaf_label_success_prob))

        print(f"{n_samples}, {leaf_label_success_prob}: {required_leaf_epsilon:.3f}")
