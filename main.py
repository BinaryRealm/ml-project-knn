import numpy as np
from knn import knn, knn_learn
from missing_values import fill_missing_values
from accuracy import check_accuracy
from column_mean import fill_with_column_mean


####DATASETS 1&2 (missing values)
"""
missing_data1 = np.loadtxt(open("MissingData1.txt", "rb"), delimiter="\t")
missing_data2 = np.loadtxt(open("MissingData2.txt", "rb"), delimiter="\t")

# fill missing values in MissingData1.txt
filled_missing_data1 = fill_missing_values(
    train_data=missing_data1, k=3, missing_value=1.0e99
)

np.savetxt(
    "GordonMissingResult1.txt", filled_missing_data1, delimiter="\t", fmt="%.15f"
)


# fill missing values in MissingData2.txt
filled_missing_data2 = fill_missing_values(
    train_data=missing_data2, k=3, missing_value=1.0e99
)

np.savetxt(
    "GordonMissingResult2.txt", filled_missing_data2, delimiter="\t", fmt="%.15f"
)
"""

####DATASET 1

"""
test_data1 = np.loadtxt(open("TestData1.txt", "rb"), delimiter="\t")
train_data1 = np.loadtxt(open("TrainData1.txt", "rb"), delimiter="\t")
train_labels1 = np.loadtxt(open("TrainLabel1.txt", "rb"), delimiter="\r")


# fill missing values with column means in train_data1 and test_data1

filled_train_data1 = fill_with_column_mean(train_data1, 1.0e99)
np.savetxt(
    "TrainData1_filled_values.txt", filled_train_data1, delimiter="\t", fmt="%.14f"
)

filled_test_data1 = fill_with_column_mean(test_data1, 1.0e99)
np.savetxt(
    "TestData1_filled_values.txt", filled_test_data1, delimiter="\t", fmt="%.14f"
)
"""

"""
test_data1 = np.loadtxt(open("TestData1_filled_values.txt", "rb"), delimiter="\t")
train_data1 = np.loadtxt(open("TrainData1_filled_values.txt", "rb"), delimiter="\t")
train_labels1 = np.loadtxt(open("TrainLabel1.txt", "rb"), delimiter="\r")

labels1 = [1, 2, 3, 4, 5]


X_train1 = train_data1[0:120]
X_test1 = train_data1[120:150]
y_train1 = train_labels1[0:120]
y_test1 = train_labels1[120:150]  # to check accuracy

predictions1 = knn_learn(
    X_train=X_train1, y_train=y_train1, X_test=X_test1, classes=labels1, k=2
)
predictions1 = np.array(predictions1)
np.savetxt("TrainLabel1_predictions.txt", predictions1, delimiter="\r", fmt="%d")
np.savetxt("TrainLabel1_predictions_comp.txt", y_test1, delimiter="\r", fmt="%d")


predictions = np.loadtxt(open("TrainLabel1_predictions.txt", "rb"), delimiter="\r")
train_labels = np.loadtxt(
    open("TrainLabel1_predictions_comp.txt", "rb"), delimiter="\r"
)
print(check_accuracy(predictions, train_labels))
"""

"""
final_predictions1 = knn_learn(
    X_train=train_data1, y_train=train_labels1, X_test=test_data1, classes=labels1, k=2
)
final_predictions1 = np.array(final_predictions1)
np.savetxt("GordonClassification1.txt", final_predictions1, delimiter="\r", fmt="%d")
"""

####DATASET 2

"""
test_data2 = np.loadtxt(open("TestData2.txt", "rb"))
train_data2 = np.loadtxt(open("TrainData2.txt", "rb"))
train_labels2 = np.loadtxt(open("TrainLabel2.txt", "rb"), delimiter="\r")
labels2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


X_train2 = train_data2[0:80]
X_test2 = train_data2[80:100]
y_train2 = train_labels2[0:80]
y_test2 = train_labels2[80:100]  # to check accuracy

predictions2 = knn_learn(
    X_train=X_train2, y_train=y_train2, X_test=X_test2, classes=labels2, k=5
)
predictions2 = np.array(predictions2)
np.savetxt("TrainLabel2_predictions.txt", predictions2, delimiter="\r", fmt="%d")
np.savetxt("TrainLabel2_predictions_comp.txt", y_test2, delimiter="\r", fmt="%d")


predictions = np.loadtxt(open("TrainLabel2_predictions.txt", "rb"), delimiter="\r")
train_labels = np.loadtxt(
    open("TrainLabel2_predictions_comp.txt", "rb"), delimiter="\r"
)
print(check_accuracy(predictions, train_labels))
"""

"""

final_predictions2 = knn_learn(
    X_train=train_data2, y_train=train_labels2, X_test=test_data2, classes=labels2, k=5
)
final_predictions2 = np.array(final_predictions2)
np.savetxt("GordonClassification2.txt", final_predictions2, delimiter="\r", fmt="%d")
"""

####DATASET 3


# correct incorrect newlines in TestData3.txt
# with open("TestData3.txt", "r") as file:
#    test_data = file.read().replace("\r", "\n")
# with open("TestData3_corrected.txt", "w") as file:
#    file.write(test_data)

"""
test_data3 = np.loadtxt(open("TestData3_corrected.txt", "rb"), delimiter=",")
train_data3 = np.loadtxt(open("TrainData3.txt", "rb"), delimiter="\t")
train_labels3 = np.loadtxt(open("TrainLabel3.txt", "rb"), delimiter="\r")
labels3 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
"""

# fill missing values in TrainData3.txt
"""
filled_train_data3 = fill_missing_values(
    train_data=train_data3, k=5, missing_value=1.0e99
)

np.savetxt(
    "TrainData3_filled_values.txt", filled_train_data3, delimiter="\t", fmt="%.3f"
)
"""

# fill missing values in TestData3.txt
"""
filled_test_data3 = fill_missing_values(
    train_data=test_data3, k=5, missing_value=1000000000
)

np.savetxt("TestData3_filled_values.txt", filled_test_data3, delimiter="\t", fmt="%.3f")
"""
"""
test_data3 = np.loadtxt(open("TestData3_filled_values.txt", "rb"), delimiter="\t")
train_data3 = np.loadtxt(open("TrainData3_filled_values.txt", "rb"), delimiter="\t")
train_labels3 = np.loadtxt(open("TrainLabel3.txt", "rb"), delimiter="\r")
labels3 = [1, 2, 3, 4, 5, 6, 7, 8, 9]


X_train3 = train_data3[0:5040]
X_test3 = train_data3[5040:6300]
y_train3 = train_labels3[0:5040]
y_test3 = train_labels3[5040:6300]  # to check accuracy

predictions = knn_learn(
    X_train=X_train3, y_train=y_train3, X_test=X_test3, classes=labels3, k=4
)
predictions = np.array(predictions)
np.savetxt("TrainLabel3_predictions.txt", predictions, delimiter="\r", fmt="%d")
np.savetxt("TrainLabel3_predictions_comp.txt", y_test3, delimiter="\r", fmt="%d")

predictions = np.loadtxt(open("TrainLabel3_predictions.txt", "rb"), delimiter="\r")
train_labels = np.loadtxt(
    open("TrainLabel3_predictions_comp.txt", "rb"), delimiter="\r"
)
print(check_accuracy(predictions, train_labels))
"""

"""
final_predictions3 = knn_learn(
    X_train=train_data3, y_train=train_labels3, X_test=test_data3, classes=labels3, k=4
)
final_predictions3 = np.array(final_predictions3)
np.savetxt("GordonClassification3.txt", final_predictions3, delimiter="\r", fmt="%d")
"""

####DATASET 4

"""
test_data4 = np.loadtxt(open("TestData4.txt", "rb"))
train_data4 = np.loadtxt(open("TrainData4.txt", "rb"))
train_labels4 = np.loadtxt(open("TrainLabel4.txt", "rb"), delimiter="\r")
labels4 = [1, 2, 3, 4, 5, 6, 7, 8, 9]


X_train4 = train_data4[0:874]
X_test4 = train_data4[874:1092]
y_train4 = train_labels4[0:874]
y_test4 = train_labels4[874:1092]  # to check accuracy

predictions4 = knn_learn(
    X_train=X_train4, y_train=y_train4, X_test=X_test4, classes=labels4, k=3
)
predictions4 = np.array(predictions4)
np.savetxt("TrainLabel4_predictions.txt", predictions4, delimiter="\r", fmt="%d")
np.savetxt("TrainLabel4_predictions_comp.txt", y_test4, delimiter="\r", fmt="%d")


predictions = np.loadtxt(open("TrainLabel4_predictions.txt", "rb"), delimiter="\r")
train_labels = np.loadtxt(
    open("TrainLabel4_predictions_comp.txt", "rb"), delimiter="\r"
)
print(check_accuracy(predictions, train_labels))
"""

"""
final_predictions4 = knn_learn(
    X_train=train_data4, y_train=train_labels4, X_test=test_data4, classes=labels4, k=3
)
final_predictions4 = np.array(final_predictions4)
np.savetxt("GordonClassification4.txt", final_predictions4, delimiter="\r", fmt="%d")
"""


####DATASET 5

"""
test_data5 = np.loadtxt(open("TestData5.txt", "rb"))
train_data5 = np.loadtxt(open("TrainData5.txt", "rb"))
train_labels5 = np.loadtxt(open("TrainLabel5.txt", "rb"), delimiter="\r")
labels5 = [3, 4, 5, 6, 7, 8]


X_train5 = train_data5[0:384]
X_test5 = train_data5[384:480]
y_train5 = train_labels5[0:384]
y_test5 = train_labels5[384:480]  # to check accuracy

predictions5 = knn_learn(
    X_train=X_train5, y_train=y_train5, X_test=X_test5, classes=labels5, k=5
)
predictions5 = np.array(predictions5)
np.savetxt("TrainLabel5_predictions.txt", predictions5, delimiter="\r", fmt="%d")
np.savetxt("TrainLabel5_predictions_comp.txt", y_test5, delimiter="\r", fmt="%d")


predictions = np.loadtxt(open("TrainLabel5_predictions.txt", "rb"), delimiter="\r")
train_labels = np.loadtxt(
    open("TrainLabel5_predictions_comp.txt", "rb"), delimiter="\r"
)
print(check_accuracy(predictions, train_labels))
"""

"""

final_predictions5 = knn_learn(
    X_train=train_data5, y_train=train_labels5, X_test=test_data5, classes=labels5, k=5
)
final_predictions5 = np.array(final_predictions5)
np.savetxt("GordonClassification5.txt", final_predictions5, delimiter="\r", fmt="%d")
"""
