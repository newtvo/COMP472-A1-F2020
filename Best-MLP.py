from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore","", ConvergenceWarning)

from sklearn.metrics import confusion_matrix, classification_report


from Dataset.dataset import *
from Output.output import *

# Creating lists of parameter for Decision Tree Classifier
Best_MLP = MLPClassifier(max_iter=100)

param_grid = [{'activation': ['identity', 'relu','tanh','logistic'], 'hidden_layer_sizes':[(150,100), (50,50)], 'solver':['adam', 'sgd']}]

# finding the best hyperparameter using gridsearchCV
grid_search = GridSearchCV(Best_MLP, param_grid, n_jobs=-1, cv=2)
grid_search.fit(feature_1, target_1)
best_param = grid_search.best_params_
print(best_param)

# Using the best parameter for the model
Best_MLP = MLPClassifier(activation='tanh',
                    hidden_layer_sizes=(150, 100),
                    solver='adam')

Best_MLP = Best_MLP.fit(feature_1, target_1)


# # apply VALIDATION SET to test first
validation_predict_1 = Best_MLP.predict(validation_feature_1)

# confusion matrix
val_confusion_matrix_1 = confusion_matrix(validation_target_1, validation_predict_1)
# precision, recall, f1-measure (macro and weighted) and accuracy
val_report_1 = classification_report(validation_target_1, validation_predict_1, target_names=label_1)
# check validation metrics
print(val_confusion_matrix_1)
print(val_report_1)

# apply for TEST SET (test_with_labels)
test_predict_1 = Best_MLP.predict(test_feature_lb_1).astype(int)

test_confusion_matrix_1 = confusion_matrix(test_target_lb_1, test_predict_1)
test_report_1 = classification_report(test_target_lb_1, test_predict_1, target_names=label_1)
# check test metrics
print(test_confusion_matrix_1)
print(test_report_1)

output_file(test_target_lb_1, test_predict_1, label_1, 'Best-MLP-DS1')

###################################################################################################

# train Gaussian NB Model for dataset 1
Best_MLP.fit(feature_2, target_2)

# # apply VALIDATION SET to test first
validation_predict_2 = Best_MLP.predict(validation_feature_2)

# confusion matrix
val_confusion_matrix_2 = confusion_matrix(validation_target_2, validation_predict_2)
# precision, recall, f1-measure (macro and weighted) and accuracy
val_report_2 = classification_report(validation_target_2, validation_predict_2, target_names=label_2)
# check validation metrics
print(val_confusion_matrix_2)
print(val_report_2)

# apply for TEST SET (test_with_labels)
test_predict_2 = Best_MLP.predict(test_feature_lb_2).astype(int)

test_confusion_matrix_2 = confusion_matrix(test_target_lb_2, test_predict_2)
test_report_2 = classification_report(test_target_lb_2, test_predict_2, target_names=label_2)
# check test metrics
print(test_confusion_matrix_2)
print(test_report_2)

# write test results to output Best_MLP-DS2.csv
output_file(test_target_lb_2, test_predict_2, label_2, 'Best-MLP-DS2')
