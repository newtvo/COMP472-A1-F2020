import numpy as np
import matplotlib.pyplot as plt

# For dataset 1 (Latin alphabet), load class indexes and label
data_index_1 = np.loadtxt('./Dataset/Dataset1/info_1.csv', skiprows=1, usecols=0, delimiter=',', dtype=np.int32)
data_label_1 = np.loadtxt('./Dataset/Dataset1/info_1.csv', skiprows=1, usecols=1, delimiter=',', dtype=np.str)

# For dataset 2 (Greek alphabet), load class indexes and label 
data_index_2 = np.loadtxt('./Dataset/Dataset2/info_2.csv', skiprows=1, usecols=0, delimiter=',', dtype=np.int32)
data_label_2 = np.loadtxt('./Dataset/Dataset2/info_2.csv', skiprows=1, usecols=1, delimiter=',', dtype=np.str)


# In a given dataset, function is being called to calculate the number of occurrences of a class
#  data_index: index array which represents the indexes of the dataset's class
#  fileName: name of the csv file
def dist_calculation(data_index, fileName, nb_pixels=32**2):
    distribution = np.zeros(data_index.shape[0], dtype=np.int32)
    data_class = np.loadtxt(fileName, usecols=nb_pixels, delimiter=',', dtype=np.int32)
    for i in data_class:
        distribution[i] += 1
    return distribution

# Distribution for Dataset 1 (Training, Validation, and Test)
data_training_1 = dist_calculation(data_index_1, './Dataset/Dataset1/train_1.csv')
data_validation_1 = dist_calculation(data_index_1, './Dataset/Dataset1/val_1.csv')
data_test_1 = dist_calculation(data_index_1, './Dataset/Dataset1/test_with_label_1.csv')

# Distribution for Dataset 2 (Training, Validation, and Test)
data_training_2 = dist_calculation(data_index_2, './Dataset/Dataset2/train_2.csv')
data_validation_2 = dist_calculation(data_index_2, './Dataset/Dataset2/val_2.csv')
data_test_2 = dist_calculation(data_index_2, './Dataset/Dataset2/test_with_label_2.csv')

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
# class_labels: Array of class labels of the dataset
# distributions: Array has length of 3 - 1D array for all 3 training, validation and test in order
# title: Title of the bar chart
# width: Width of a single bar
# font_size: Font size to be appeared on the graph of each bar
def plot_class_dist(indexes, class_labels, distributions, title, width=0.25, font_size='xx-small'):
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
    ax.set_xticklabels(class_labels)
    ax.legend()

    fig.tight_layout()

    plt.show()

# Plot each distributions
def plot_dist():

    plot_class_dist(data_index_1, data_label_1, [data_training_1, data_validation_1, data_test_1], 'Distribution of the number of the instances in Dataset 1')

    plot_class_dist(data_index_2, data_label_2, [data_training_2, data_validation_2, data_test_2], 'Distribution of the number of the instances in Dataset 2')