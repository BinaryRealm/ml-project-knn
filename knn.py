"""
K Nearest Neighbors
"""
import math
import numpy as np


def knn(train_data, test_data, k, train_labels=None, classes=None, missing_value=None):
    """
    knn algo
    """
    ignore_indexes = []
    if missing_value is not None:
        for i in range(len(test_data)):
            if test_data[i] == missing_value:
                ignore_indexes.append(i)

    diffs = []
    for i in range(len(train_data)):
        diff = []
        for j in range(len(train_data[i])):
            if j in ignore_indexes:
                continue
            diff.append(abs((train_data[i][j] - test_data[j])))
        diffs.append(diff)

    distances = []
    for i in range(len(diffs)):
        sum = 0
        for j in range(len(diffs[0])):
            sum += diffs[i][j] * diffs[i][j]
        distances.append(math.sqrt(sum))
    # print(distances)

    indexes = []
    for j in range(k):
        index = -1
        min = 100000000
        for i in range(len(distances)):
            if i in indexes:
                continue
            if distances[i] < min:
                min = distances[i]
                index = i
        indexes.append(index)

    # print(f"closest rows in train data: {indexes}")

    if (
        train_labels is None or classes is None
    ):  # if predictions are not being made based on training labels
        return indexes, ignore_indexes

    class_counts = np.zeros(len(classes))

    for i in range(len(indexes)):
        for j in range(len(classes)):
            if train_labels[indexes[i]] == classes[j]:
                class_counts[j] += 1

    # print(f"frequency of label classes: {class_counts}")
    max_val = np.amax(class_counts)
    index = np.where(class_counts == max_val)[0][0]
    return classes[index], indexes, ignore_indexes


def knn_learn(X_train, y_train, X_test, classes, k):
    """
    knn_learn
    """
    predictions = []
    for row in X_test:
        prediction, indexes, ignore_indexes = knn(
            train_data=X_train,
            test_data=row,
            k=k,
            train_labels=y_train,
            classes=classes,
            missing_value=None,
        )
        predictions.append(prediction)
    return np.array(predictions)


"""
data = [
    [1.0, 2.3, 5.2, 1.2, 5.3, 2.6, 2.3],
    [2.0, 3.6, 1.8, 2.3, 1.6, 2.1, 1.5],
    [1.5, 1.5, 4.1, 1.3, 1.2, 3.1, 1.6],
    [2.2, 1.9, 9.5, 1.5, 1.5, 4.2, 1.4],
    [3.9, 2.4, 5.3, 1.7, 1.6, 2.5, 2.9],
    [5.1, 3.6, 2.7, 2.6, 1.7, 2.8, 3.4],
    [1.8, 4.2, 3.6, 3.5, 1.6, 3.4, 1.3],
    [2.3, 1.5, 7.2, 4.1, 7.1, 3.1, 1.8],
    [4.2, 2.4, 6.2, 2.9, 2.5, 3.3, 2.5],
    [3.6, 5.6, 1.9, 3.2, 2.6, 5.2, 2.7],
]
data = np.array(data)
data_outputs = ["Yes", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No"]
data_outputs = np.array(data_outputs)

sample11 = np.array([2.1, 2.2, 3.2, 1.4, 5.1, 2.4, 1.4])
sample12 = np.array([2.4, 2.3, 3.4, 3.8, 2.3, 5.7, 5.2])
my_sample = np.array([1.0, 2.31, 5.2, 1.2, 5.3, 2.6, 2.3])
my_sample2 = np.array([2.1, 3.6, 1.8, 2.3, 1.6, 2.1, 1.5])
my_sample3 = np.array([2.0, 1.0e99, 1.0e99, 1.0e99, 1.0e99, 1.0e99, 1.0e99])
classes = ["Yes", "No"]

print(
    f"my_sample1: {knn(train_data=data, train_labels=data_outputs, test_data=my_sample, k=1, classes=classes)}\n"
)
print(
    f"my_sample2: {knn(train_data=data, train_labels=data_outputs, test_data=my_sample2, k=1, classes=classes)}\n"
)
print(
    f"my_sample3: {knn(train_data=data, train_labels=data_outputs, test_data=my_sample3, k=3, classes=classes, missing_value=1.0e99)}\n"
)
# print(f"Sample 11: {knn(data, data_outputs, sample11, 3, classes)}\n")
# print(f"Sample 12: {knn(data, data_outputs, sample12, 3, classes)}\n")
"""
