import numpy as np
import openml


def print_dataset_info(dataset):
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )

    keep_rows = ~np.any(np.isnan(X), axis=1)
    X = X[keep_rows]
    y = y[keep_rows]

    _, y = np.unique(y, return_inverse=True)
    count_0, count_1 = np.bincount(y, minlength=2)
    dummy_score = max(count_0, count_1) / len(y)

    n_categorical_features = np.sum(categorical_indicator)

    print(
        dataset.name, *X.shape, n_categorical_features, f"{dummy_score:.3f}", sep=", "
    )


numerical_task_ids = openml.study.get_suite(337).tasks
categorical_task_ids = openml.study.get_suite(334).tasks
uci_data_ids = [
    15,  # breast cancer (Wisconsin)
    24,  # mushroom
    37,  # diabetes
    56,  # vote
    959,  # nursery
    1590,  # adult
]

for task_id in numerical_task_ids:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()

    print_dataset_info(dataset)

for task_id in categorical_task_ids:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()

    print_dataset_info(dataset)

for data_id in uci_data_ids:
    dataset = openml.datasets.get_dataset(data_id)

    print_dataset_info(dataset)
