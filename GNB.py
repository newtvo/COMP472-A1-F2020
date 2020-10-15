import numpy as np
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

from Dataset.dataset import *

GNB = GaussianNB()
# train Gaussian NB Model for dataset 1
GNB.fit(feature_1, target_1)

# apply validation set to test first
validation_predict_1 = GNB.predict(validation_feature_1)


# confusion matrix
confusion_matrix = confusion_matrix(validation_predict_1, validation_target_1)
classification_report = classification_report(validation_target_1, validation_predict_1, target_names=label_1)
# print(confusion_matrix)
print(classification_report)



