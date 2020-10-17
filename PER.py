from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

from Dataset.dataset import *
from Output.output import *

PER = Perceptron()
# train Gaussian NB Model for dataset 1
PER.fit(feature_1, target_1)

# # apply VALIDATION SET to test first
validation_predict_1 = PER.predict(validation_feature_1)

# confusion matrix
val_confusion_matrix_1 = confusion_matrix(validation_target_1, validation_predict_1)
# precision, recall, f1-measure (macro and weighted) and accuracy
val_report_1 = classification_report(validation_target_1, validation_predict_1, target_names=label_1)
# check validation metrics
print(val_confusion_matrix_1)
print(val_report_1)

# apply for TEST SET (test_with_labels)
test_predict_1 = PER.predict(test_feature_lb_1).astype(int)

test_confusion_matrix_1 = confusion_matrix(test_target_lb_1, test_predict_1)
test_report_1 = classification_report(test_target_lb_1, test_predict_1, target_names=label_1)
# check test metrics
print(test_confusion_matrix_1)
print(test_report_1)

output_file(test_target_lb_1, test_predict_1, label_1, 'PER-DS1')

###################################################################################################

# train Gaussian NB Model for dataset 1
PER.fit(feature_2, target_2)

# # apply VALIDATION SET to test first
validation_predict_2 = PER.predict(validation_feature_2)

# confusion matrix
val_confusion_matrix_2 = confusion_matrix(validation_target_2, validation_predict_2)
# precision, recall, f1-measure (macro and weighted) and accuracy
val_report_2 = classification_report(validation_target_2, validation_predict_2, target_names=label_2)
# check validation metrics
print(val_confusion_matrix_2)
print(val_report_2)

# apply for TEST SET (test_with_labels)
test_predict_2 = PER.predict(test_feature_lb_2).astype(int)

test_confusion_matrix_2 = confusion_matrix(test_target_lb_2, test_predict_2)
test_report_2 = classification_report(test_target_lb_2, test_predict_2, target_names=label_2)
# check test metrics
print(test_confusion_matrix_2)
print(test_report_2)

# write test results to output PER-DS2.csv
output_file(test_target_lb_2, test_predict_2, label_2, 'PER-DS2')
