import numpy as np
import matplotlib.pyplot as plt

# For dataset 1 (Latin alphabet), load class indexes and label
dataIndex1 = np.loadtxt('./Dataset/Dataset1/info_1.csv', skiprows=1, usecols=0, delimiter=',', dtype=np.int32)
dataLabels1 = np.loadtxt('./Dataset/Dataset1/info_1.csv', skiprows=1, usecols=1, delimiter=',', dtype=np.str)

# For dataset 2 (Greek alphabet), load class indexes and label 
dataIndex2 = np.loadtxt('./Dataset/Dataset2/info_2.csv', skiprows=1, usecols=0, delimiter=',', dtype=np.int32)
dataLabels2 = np.loadtxt('./Dataset/Dataset2/info_2.csv', skiprows=1, usecols=1, delimiter=',', dtype=np.str)


# In a given dataset, function is being called to calculate the number of occurrences of a class
#  dataIndex: index array which represents the indexes of the dataset's class
#  fileName: name of the csv file
def dist_calculation(dataIndex, fileName, nb_pixels=32**2):
    distribution = np.zeros(dataIndex.shape[0], dtype=np.int32)
    dataClass = np.loadtxt(fileName, usecols=nb_pixels, delimiter=',', dtype=np.int32)
    for i in dataClass:
        distribution[i] += 1
    return distribution

# Distribution for Dataset 1 (Training, Validation, and Test)
dataTraining1 = dist_calculation(dataIndex1, './Dataset/Dataset1/train_1.csv')
dataValidation1 = dist_calculation(dataIndex1, './Dataset/Dataset1/val_1.csv')
dataTest1 = dist_calculation(dataIndex1, './Dataset/Dataset1/test_with_label_1.csv')

# Distribution for Dataset 2 (Training, Validation, and Test)
dataTraining2 = dist_calculation(dataIndex2, './Dataset/Dataset2/train_2.csv')
dataValidation2 = dist_calculation(dataIndex2, './Dataset/Dataset2/val_2.csv')
dataTest2 = dist_calculation(dataIndex2, './Dataset/Dataset2/test_with_label_2.csv')

# Function used to plot bar graph distribution
def plot_bar_graph(rects, ax):
    # Set a text label for each bar to display its height
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize='xx-small')


# Function to plot the distributions of each dataset by class labels
#
# indexes: Array of class indexes of the dataset
# classLabels: Array of class labels of the dataset
# distributions: Array has length of 3 - 1D array for all 3 training, validation and test in order
# title: Title of the bar chart
# width: Width of a single bar
# font_size: Font size to be appeared on the graph of each bar
def plot_class_dist(indexes, classLabels, distributions, title, width=0.25, font_size='xx-small'):
    # Bar generation
    fig, ax = plt.subplots()
    rects_training = ax.bar(indexes - width, distributions[0], width, label='Training')
    rects_val = ax.bar(indexes, distributions[1], width, label='Validation')
    rects_test = ax.bar(indexes + width, distributions[2], width, label='Test')

    # Counting height for each bar
    plot_bar_graph(rects_training, ax)
    plot_bar_graph(rects_val, ax)
    plot_bar_graph(rects_test, ax)

    # Bar chart configuration
    ax.grid(b=True, axis='y')
    ax.set_title(title)
    ax.set_xlabel('Character')
    ax.set_ylabel('Count')
    ax.set_xticks(indexes)
    ax.set_xticklabels(classLabels)
    ax.legend()

    fig.tight_layout()

    plt.show()

# Plot each distributions
def plot_dist():

    plot_class_dist(dataIndex1, dataLabels1, [dataTraining1, dataValidation1, dataTest1], 'Distribution of the number of the instances in Dataset 1')

    plot_class_dist(dataIndex2, dataLabels2, [dataTraining2, dataValidation2, dataTest2], 'Distribution of the number of the instances in Dataset 2')