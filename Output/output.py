from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


def output_file(target_val, predicted_target_val, labels, file_name):
    # read file
    output = open('Output/' + file_name + '.csv', 'w')
    # write value of test data
    output.write('instance, prediction\n')

    # loop through the numpy array and write the file
    # with 1D dimension - n rows
    for i in range(predicted_target_val.shape[0]):
        output.write(str(i+1) + ',' + str(predicted_target_val[i]) + '\n')

    output.write('\n')

    # plot confusion matrix
    output.write('confusion matrix\n')
    matrix = confusion_matrix(target_val, predicted_target_val)
    (m, n) = matrix.shape
    for i in range(m):
        for j in range(n):
            if j < n - 1:
                output.write(str(matrix[i, j]) + ',')
            else:
                output.write(str(matrix[i, j]))
        output.write('\n')

    output.write('\n')

    # write precision, recall, and f1-measure for each class
    output.write('precision, recall, f1-measure\n')
    precision = precision_score(target_val, predicted_target_val, average=None)
    recall = recall_score(target_val, predicted_target_val, average=None)
    f1 = f1_score(target_val, predicted_target_val, average=None)

    # round to 2 decimals
    for i in range(labels.shape[0]):
        output.write('{:.2f},{:.2f},{:.2f}\n'.format(precision[i], recall[i], f1[i]))

    output.write('\n')

    # write accuracy, marco-average f1 and weighted-average f1
    output.write('accuracy, macro-average f-1, weighted-average f1\n')
    accuracy = accuracy_score(target_val, predicted_target_val)
    macro_avg_f1 = f1_score(target_val, predicted_target_val, average='macro')
    weighted_avg_f1 = f1_score(target_val, predicted_target_val, average='weighted')

    output.write('{:.2f},{:.2f},{:.2f}\n'.format(accuracy, macro_avg_f1, weighted_avg_f1))

    # close output file
    output.close()


