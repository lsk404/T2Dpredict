from pytorch_tabnet.tab_model import TabNetRegressor,TabNetClassifier
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
# X_train = pandas.read_csv()
train = np.load("data/train.npy")
valid = np.load("data/valid.npy")
test = np.load("data/test.npy")
X_train = train[:,:-1]
Y_train = train[:,-1].reshape(-1,1)
X_valid = valid[:,:-1]
Y_valid = valid[:,-1].reshape(-1,1)
X_test = test[:,:-1]
Y_test = test[:,-1].reshape(-1,1)
import sys,time
tik = time.time()
localtime = time.localtime()
mday = localtime.tm_mday
h = localtime.tm_hour
m = localtime.tm_min
sys.stdout = open(f"log{mday:02d}{h:02d}{m:02d}.txt", "w+")
clf = TabNetRegressor(
  gamma=0.8,
  optimizer_fn=torch.optim.Adam,
  optimizer_params=dict(lr=1e-2)
)
from pytorch_tabnet.augmentations import RegressionSMOTE
aug = RegressionSMOTE(p=0.8)
clf.fit(
   X_train=X_train, y_train=Y_train,
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
    max_epochs=1000,
    patience=50,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    augmentations=aug, #aug
)
preds = clf.predict(X_test)
print(preds.shape)
print(Y_test.shape)
print("=====preds=====")
print(preds)
print("=====preds - Y_test=====")
print(preds-Y_test)
print("======para=====")
print(clf.get_params())
print("======scores=====")

np.save("preds.npy",preds)
tok = time.time()
print(tok-tik)
np.save("feature_importance_.npy",clf.feature_importances_)

