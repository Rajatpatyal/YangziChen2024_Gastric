# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

def plot_matrix_2d(y_true, y_pred, labels_name, out_dir_path, title=None, thresh=0.8, axis_labels=None):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  

    pl.imshow(cm * 100, interpolation='nearest', cmap=pl.get_cmap('Reds'))
    pl.clim(0, 100)
    pl.colorbar()  

    if title is not None:
        pl.title(title)

    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels, rotation=45)  
    pl.yticks(num_local, axis_labels)  
    pl.ylabel('True label')
    pl.xlabel('Predicted label')

    pl.text(0, 0, "TN:" + format(round(cm[0][0] * 100, 2), '.2f') + '%',
                        ha="center", va="center",
                        color="white" if cm[0][0] > thresh else "black")
    pl.text(0, 1, "FN:" + format(round(cm[1][0] * 100, 2), '.2f') + '%',
                    ha="center", va="center",
                    color="white" if cm[1][0] > thresh else "black")

    pl.text(1, 0, "FP:" + format(round(cm[0][1] * 100, 2), '.2f') + '%',
                    ha="center", va="center",
                    color="white" if cm[0][1] > thresh else "black")
    pl.text(1, 1, "TP:" + format(round(cm[1][1] * 100, 2), '.2f') + '%',
                    ha="center", va="center",
                    color="white" if cm[1][1] > thresh else "black")
    
    pl.savefig(out_dir_path, format='pdf', bbox_inches='tight')
    pl.show()

def predict_state_value(pred_score):
    if pred_score < 0.5:
        return 'Healthy'
    else:
        return 'GC'
    
def predict_stage_value(row):
    if row['predict_state'] == 'Healthy':
        return 'N'
    elif (row['predict_state'] == 'GC') & (row['stage'] != 'N'):
        return row['stage']
    elif (row['predict_state'] == 'GC') & (row['stage'] == 'N'):
        return row['predict_state']

def calculate_recall_precision(true_labels, predicted_labels, positive_label='GC', negative_label='Healthy'):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == positive_label and predicted_label == positive_label:
            true_positives += 1
        elif true_label == negative_label and predicted_label == positive_label:
            false_positives += 1
        elif true_label == positive_label and predicted_label == negative_label:
            false_negatives += 1
    print(true_positives)
    print(false_negatives)
    print(false_positives)

    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    return recall, precision

# Read model prediction result for test dataset.
test = pd.read_excel('data/model_prediction_results_for_test.xlsx')

test['predict_state'] = test['pred_score'].apply(predict_state_value)
test_stage_IA = test[(test['stage'] == 'N') | (test['stage'] == 'IA')]
test_stage_IB = test[(test['stage'] == 'N') | (test['stage'] == 'IB')]
test_stage_II = test[(test['stage'] == 'N') | (test['stage'] == 'II')]
test_stage_III = test[(test['stage'] == 'N') | (test['stage'] == 'III')]
test_stage_IV = test[(test['stage'] == 'N') | (test['stage'] == 'IV')]

label_test = list(test.clinical_state)
label_test_predict = list(test.predict_state)
plot_matrix_2d(label_test, label_test_predict, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_test_for_all_stage.pdf")

label_test_stage_IA = list(test_stage_IA.clinical_state)
label_test_predict_stage_IA = list(test_stage_IA.predict_state)
plot_matrix_2d(label_test_stage_IA, label_test_predict_stage_IA, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_test_for_stage_IA.pdf")

label_test_stage_IB = list(test_stage_IB.clinical_state)
label_test_predict_stage_IB = list(test_stage_IB.predict_state)
plot_matrix_2d(label_test_stage_IB, label_test_predict_stage_IB, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_test_for_stage_IB.pdf")

label_test_stage_II = list(test_stage_II.clinical_state)
label_test_predict_stage_II = list(test_stage_II.predict_state)
plot_matrix_2d(label_test_stage_II, label_test_predict_stage_II, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_test_for_stage_II.pdf")

label_test_stage_III = list(test_stage_III.clinical_state)
label_test_predict_stage_III = list(test_stage_III.predict_state)
plot_matrix_2d(label_test_stage_III, label_test_predict_stage_III, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_test_for_stage_III.pdf")

label_test_stage_IV = list(test_stage_IV.clinical_state)
label_test_predict_stage_IV = list(test_stage_IV.predict_state)
plot_matrix_2d(label_test_stage_IV, label_test_predict_stage_IV, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_test_for_stage_IV.pdf")

# Read model prediction result for external test dataset.
external_test = pd.read_excel('data/model_prediction_results_for_external_test.xlsx')

external_test['predict_state'] = external_test['pred_score'].apply(predict_state_value)

external_test_stage_IA = external_test[(external_test['stage'] == 'N') | (external_test['stage'] == 'IA')]
external_test_stage_IB = external_test[(external_test['stage'] == 'N') | (external_test['stage'] == 'IB')]
external_test_stage_II = external_test[(external_test['stage'] == 'N') | (external_test['stage'] == 'II')]
external_test_stage_III = external_test[(external_test['stage'] == 'N') | (external_test['stage'] == 'III')]
external_test_stage_IV = external_test[(external_test['stage'] == 'N') | (external_test['stage'] == 'IV')]

label_external_test = list(external_test.clinical_state)
label_external_test_predict = list(external_test.predict_state)
plot_matrix_2d(label_external_test, label_external_test_predict, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_external_test_for_all_stage.pdf")

label_external_test_stage_IA = list(external_test_stage_IA.clinical_state)
label_external_test_predict_stage_IA = list(external_test_stage_IA.predict_state)
plot_matrix_2d(label_external_test_stage_IA, label_external_test_predict_stage_IA, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_external_test_for_stage_IA.pdf")

label_external_test_stage_IB = list(external_test_stage_IB.clinical_state)
label_external_test_predict_stage_IB = list(external_test_stage_IB.predict_state)
plot_matrix_2d(label_external_test_stage_IB, label_external_test_predict_stage_IB, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_external_test_for_stage_IB.pdf")

label_external_test_stage_II = list(external_test_stage_II.clinical_state)
label_external_test_predict_stage_II = list(external_test_stage_II.predict_state)
plot_matrix_2d(label_external_test_stage_II, label_external_test_predict_stage_II, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_external_test_for_stage_II.pdf")

label_external_test_stage_III = list(external_test_stage_III.clinical_state)
label_external_test_predict_stage_III = list(external_test_stage_III.predict_state)
plot_matrix_2d(label_external_test_stage_III, label_external_test_predict_stage_III, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_external_test_for_stage_III.pdf")

label_external_test_stage_IV = list(external_test_stage_IV.clinical_state)
label_external_test_predict_stage_IV = list(external_test_stage_IV.predict_state)
plot_matrix_2d(label_external_test_stage_IV, label_external_test_predict_stage_IV, ['Healthy', 'GC'], title='confusion matrix',
            axis_labels=['Healthy', 'GC'], out_dir_path="figures/Confusion_matrix/confusion_matrix_external_test_for_stage_IV.pdf")

print('Done')