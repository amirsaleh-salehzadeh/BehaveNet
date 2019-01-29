import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random as rn

from sklearn.metrics.classification import confusion_matrix, \
    precision_recall_fscore_support
import itertools
from sklearn.metrics.ranking import roc_curve, auc
from scipy import interp
from itertools import cycle
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

rn.seed(12345)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(3)
rn.seed(12345)

folder = "res/"


def compute_accuracy(name, predictions, y_labels, history):
    drawPLT(history, name)
    predicted_labels = []
    for prediction in predictions:
        prediction_list = list(prediction)
        predicted_labels.append(prediction_list.index(max(prediction_list)))
    correct = 0
    for label in range(len(predicted_labels)):
        if predicted_labels[label] == y_labels[label]:
            correct += 1
    accuracy = 100 * (correct / len(predicted_labels))
    tttt = "\n{},{},{},{}%".format(name,correct, len(predicted_labels), accuracy)
    f = open(folder + name + "predAcc.csv", "a")
    f.write(tttt)
    f.close()
#     np.savetxt('res/pred.csv', predictions, delimiter=',')
    return


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap('PuRd'), 
                          name=""):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("res/{}MTRX".format(name))
    plt.close()

    
def evalRes(pred, test_labels, y_testMultiClass, name):
    y_pred = np.argmax(pred, axis=1)
    y_test = test_labels
    target_names = ['Reading', 'Speaking', 'Watching']
    cnf_matrix = confusion_matrix(y_pred, test_labels)
    df_class_report = pandas_classification_report(y_true=y_test, y_pred=y_pred)
    df_class_report.to_csv(folder + name+ 'classification_report.csv', sep=',')
    plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                          title='', name=name)
    plot_ROC(pred, y_testMultiClass, name)


def plot_ROC(y_pred, y_test, name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 3
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    target_names = ['Reading', 'Speaking', 'Watching']
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC (AUC = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC (AUC = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC - {0} (AUC = {1:0.2f})'
                 ''.format(target_names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("res/{}-ROC-AUC".format(name))


def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-1] = total

    class_report_df['avg / total'] = avg
    
    return class_report_df.T

def drawPLT(history, name):
    plt.figure(1)  
    plt.subplot(211)  
    plt.plot(history.history['acc'])  
    plt.plot(history.history['val_acc'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    
    plt.subplot(212)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left') 
    plt.subplots_adjust(hspace=0.5) 
    plt.savefig("res/{}cc_loss".format(name))
    plt.close()