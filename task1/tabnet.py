from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import numpy
import pandas
from sklearn import metrics
# X_train = pandas.read_csv()
X_train = numpy.load("X_train_smote.npy")
Y_train = numpy.load("Y_train_smote.npy").reshape(-1)
X_valid = numpy.load("X_test.npy")
y_valid = numpy.load("Y_test.npy").reshape(-1)
X_test = numpy.load("X_test.npy")
Y_test = numpy.load("Y_test.npy").reshape(-1)

# exit(0)

TabNetPretrainer()
tabnet_params = {
                 "optimizer_fn":torch.optim.Adam,
                 "optimizer_params":dict(lr=2e-2),
                 "scheduler_params":{"step_size":3, # how to use learning rate scheduler
                                 "gamma":0.9},
                 "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                 "mask_type":'entmax', # "sparsemax"
                }

clf = TabNetClassifier(**tabnet_params
                      )

clf.fit(
  X_train, Y_train,
  eval_set=[(X_valid, y_valid)],
  max_epochs=100,
  eval_metric=['auc'],
  compute_importance=True,
  patience=30
)
preds = clf.predict(X_test)
# print(preds)

def precision_recall_thershold(pred_proba, y_test):
    t_recall_nodiab, t_recall_diab = [], []
    t_precision_nodiab, t_precision_diab = [], []

    for thresh in numpy.arange(0, 1, 0.01):
        precision, recall, fscore, support = \
                metrics.precision_recall_fscore_support(
                        y_test,
                        numpy.where(pred_proba[:] > thresh, 0, 1))
        recall_nodiab, recall_diab = recall
        precision_nodiab, precision_diab = precision

        t_recall_nodiab.append(recall_nodiab)
        t_recall_diab.append(recall_diab)

        t_precision_nodiab.append(precision_nodiab)
        t_precision_diab.append(precision_diab)

    return t_precision_nodiab, t_precision_diab, \
          t_recall_nodiab, t_recall_diab

a,b,c,d = precision_recall_thershold(preds, Y_test)

print(numpy.sum(preds))
print(numpy.sum(Y_test))

print(numpy.sum(preds == Y_test))
