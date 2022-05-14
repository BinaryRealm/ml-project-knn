def fill_with_column_mean(data, missing_val):
    """
    fill missing values with the mean of each column
    """
    avgs = []
    for j in range(len(data[0])):
        column = []
        for i in range(len(data)):
            if data[i][j] != missing_val:
                column.append(data[i][j])

        sum = 0
        for val in column:
            sum += val
        avg = sum / len(column)
        avgs.append(avg)
    for j in range(len(data[0])):
        for i in range(len(data)):
            if data[i][j] == missing_val:
                data[i][j] = avgs[j]

    return data
