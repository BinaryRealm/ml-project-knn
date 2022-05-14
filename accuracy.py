def check_accuracy(predictions, train_labels):
    """Returns percentage similarity between two lists"""
    correct = 0
    length = len(predictions)
    for i in range(length):
        if predictions[i] == train_labels[i]:
            correct += 1
    return correct / length
