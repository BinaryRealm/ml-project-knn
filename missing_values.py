"""
Fill missing values
Using KNN imputation
"""
import numpy as np
from knn import knn


def remove_mv_rows(train_data, missing_value, label_data=None):
    """
    return new array with rows that have no missing values
    """
    new_train_data = []
    new_label_data = []

    if label_data is not None:
        for i in range(len(train_data)):
            if missing_value not in train_data[i]:
                new_train_data.append(train_data[i])
                new_label_data.append(label_data[i])

        return new_train_data, new_label_data
    else:
        for i in range(len(train_data)):
            if missing_value not in train_data[i]:
                new_train_data.append(train_data[i])
        return new_train_data


def fill_missing_values(train_data, k, missing_value):
    """
    replace missing values in rows utilizing knn
    """
    filled_train_data = np.array(train_data)
    clean_train_data = remove_mv_rows(train_data, missing_value)
    for r in range(len(train_data)):
        if missing_value in train_data[r]:
            # print(f"finding knn for {train_data[r]}")
            index, ignore_indexes = knn(
                train_data=clean_train_data,
                test_data=train_data[r],
                k=k,
                missing_value=missing_value,
            )

            # print(f"The {k} nearest neighbors:")
            # for i in index:
            # print(clean_train_data[i])
            # print(f"The missing indexes: {ignore_indexes}")

            k_similar_rows = []
            for i in index:
                k_similar_rows.append(clean_train_data[i])
            diffs = []
            for i in range(len(k_similar_rows)):
                diff = []
                for j in range(len(k_similar_rows[i])):
                    if j in ignore_indexes:
                        continue
                    diff.append(abs((k_similar_rows[i][j] - train_data[r][j])))
                diffs.append(np.array(diff))
            sums = np.zeros(len(diffs))
            for i in range(len(diffs)):
                sums[i] = np.sum(diffs[i])
            # print(f"diffs: {diffs}")
            # print(f"sums of differences: {sums}")

            sum_of_sum_reciprocals = 0
            for s in sums:
                if s == 0:
                    sum_of_sum_reciprocals += (
                        1 / 0.01
                    )  # prevent divide by zero in formula
                else:
                    sum_of_sum_reciprocals += 1 / s

            weights = np.zeros(len(diffs))
            for i in range(len(diffs)):
                if sums[i] == 0:
                    weights[i] = (1 / 0.01) / sum_of_sum_reciprocals
                else:
                    weights[i] = (1 / sums[i]) / sum_of_sum_reciprocals

            # print(f"weights: {weights}")
            missing_val = 0
            for i in range(len(ignore_indexes)):
                for j in range(len(k_similar_rows)):
                    missing_val += k_similar_rows[j][ignore_indexes[i]] * weights[j]
                filled_train_data[r][ignore_indexes[i]] = missing_val
                missing_val = 0
            # print(f"corrected row: ")
            # print(f"{filled_train_data[r]}")
            # print()
    return filled_train_data
