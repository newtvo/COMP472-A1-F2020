# # a) instance number and predicted label (number)
# def output_file(actual_val, predicted_val, labels, file_name):
#     # read file
#     output = open('Output/' + file_name + '.csv', 'w')
#     output.write('instance,prediction\n')
#
#     # loop through the numpy array and write the file
#     # with 1D dimension - n rows
#     for i in range(actual_val.shape[0]):
#         # output.write(str(i+1) + ',' + str(actual_val[i] + ','))
#         print(str(i+1) + ',' + str(actual_val[i] + ','))
#
#     print('\n')
#     output.write('\n')
