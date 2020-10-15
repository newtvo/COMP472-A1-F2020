import numpy as np

#Dataset1
label_1 = np.loadtxt("./Dataset/Dataset1/info_1.csv", skiprows=1, usecols=1, delimiter=',', dtype=np.str)
training_set_1 = np.loadtxt("./Dataset/Dataset1/train_1.csv", skiprows=1, delimiter=",")
feature_1 = training_set_1[:,:-1]
target_1 = training_set_1[:,-1]
validation_set_1 = np.loadtxt("./Dataset/Dataset1/val_1.csv", skiprows=1, delimiter=",")
validation_feature_1 = validation_set_1[:,:-1]
validation_target_1 = validation_set_1[:,-1]
test_set_1 = np.loadtxt("./Dataset/Dataset1/test_no_label_1.csv", delimiter=',', dtype=np.int32)
test_feature_1 = test_set_1[:,:-1]
test_target_1 = test_set_1[:,-1]
test_set_lb_1 = np.loadtxt("./Dataset/Dataset1/test_with_label_1.csv", delimiter=',', dtype=np.int32)
test_feature_lb_1 = test_set_lb_1[:, :-1]
test_target_lb_1 = test_set_lb_1[:, -1]

#Dataset2
label_2 = np.loadtxt("./Dataset/Dataset2/info_2.csv", skiprows=1, usecols=1, delimiter=',', dtype=np.str)
training_set_2 = np.loadtxt("./Dataset/Dataset2/train_2.csv", skiprows=1, delimiter=",")
feature_2 = training_set_2[:,:-1]
target_2 = training_set_2[:,-1]
validation_set_2 = np.loadtxt("./Dataset/Dataset2/val_2.csv", skiprows=1, delimiter=",")
validation_feature_2 = validation_set_2[:,:-1]
validation_target_2 = validation_set_2[:,-1]
test_set_2 = np.loadtxt("./Dataset/Dataset2/test_no_label_2.csv", skiprows=1, delimiter=',', dtype=np.int32)
test_feature_2 = test_set_2[:,:-1]
test_target_2 = test_set_2[:,-1]
test_set_lb_2 = np.loadtxt("./Dataset/Dataset2/test_with_label_2.csv", delimiter=',', dtype=np.int32)
test_feature_lb_2 = test_set_lb_2[:, :-1]
test_target_lb_2 = test_set_lb_2[:, -1]