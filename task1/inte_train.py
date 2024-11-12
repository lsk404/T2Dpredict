import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys 
from sklearn import metrics
import smote

# sys.path.append('/home/aistudio/external-libraries')
# sys.path.append('/home/aistudio/work')
from utils.utils import (GridSearchCVWrapper, 
                          one_hot_encode, 
                          precision_recall_thershold, 
                          plot_recall_vs_decision_boundary, 
                          plot_multi_recall_vs_decision_boundary,
                          plot_roc_curves,
                          plot_bootstrap_roc,
                          bootstrap_model, 
                          roc_interp)

# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KneightborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC,linearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

# Helper functions
from sklearn import metrics
import time
train = np.load('train.npy')
test = np.load('test.npy')

mode = 1 ## 1: Label1, 2: Label2
if(mode == 1):
    train = train[:,:-1]
    test = test[:,:-1]
    train_0 = train[train[:,-1] == 0]
    train_1 = train[train[:,-1] == 1]
    test_0 = test[test[:,-1] == 0]
    test_1 = test[test[:,-1] == 1]
elif(mode == 2):
    train = train[train[:,-2] == 1]
    train = np.hstack((train[:,:-2],train[:,-1].reshape(-1,1)))
    train_0 = train[train[:,-1] == 0]
    train_1 = train[train[:,-1] == 1]
    
    test = test[test[:,-2] == 1]
    test = np.hstack((test[:,:-2],test[:,-1].reshape(-1,1)))
    test_0 = test[test[:,-1] == 0]
    test_1 = test[test[:,-1] == 1]
train_times = round(len(train_0)/len(train_1))
test_times = round(len(test_0)/len(test_1))
## 过采样
s_train=smote.Smote(train_1,N=train_times*100)
s_test=smote.Smote(test_1,N=test_times*100)
train_1 = s_train.over_sampling()
test_1 = s_test.over_sampling()
train = np.vstack((train_0,train_1))
test = np.vstack((test_0,test_1))
X_train = train[:,:-1]
Y_train = train[:,-1].reshape(-1)
X_test = test[:,:-1]
Y_test = test[:,-1].reshape(-1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


print(Y_test)
plt.suptitle("Feature Distribution")
fea = [0,28,27,16]
fea_name = ['Age','Age at recruitment','BMI','Spirits intake']
plt.tight_layout()
fig,axes = plt.subplots(4,4)
for i in range(4):
    for j in range(4):
        ax = axes[i,j]
        T_point_X = []
        T_point_Y = []
        F_point_X = []
        F_point_Y = []
        for k in range(len(X_test)):
            if(Y_test[k]==1):
                T_point_X.append(X_test[k,fea[i]])
                T_point_Y.append(X_test[k,fea[j]])
            else:
                F_point_X.append(X_test[k,fea[i]])
                F_point_Y.append(X_test[k,fea[j]])
        ax.scatter(T_point_X,T_point_Y,color='red',label='T2D',s=5)
        ax.scatter(F_point_X,F_point_Y,color='green',label='healthy',s=5)
        ax.set_xlabel(fea_name[i])
        ax.set_ylabel(fea_name[j])

plt.legend()
plt.show()
exit()


svc_best_params, svc_best_score = GridSearchCVWrapper(
    model=SVC(), 
    param_grid=dict(
        C=[10], 
        kernel=["rbf"]
    ),
    X=X_train,
    y=Y_train
)
svc_best_params['probability'] = True

tik = time.time()
rf_best_params, rf_best_score = GridSearchCVWrapper(
    model=RandomForestClassifier(),
    param_grid=dict(
        criterion=["entropy"],
        min_samples_leaf=[15],
        min_samples_split=[2],
        max_features=["sqrt"]
    ),
    X=X_train,
    y=Y_train
)

lg_best_params, lg_best_score = GridSearchCVWrapper(
    model=LogisticRegression(),
    param_grid=dict(
        penalty=["l1"],
        solver=['liblinear']
    ),
    X=X_train,
    y=Y_train
)

knn_best_params, knn_best_score = GridSearchCVWrapper(
    model=KNeighborsClassifier(),
    param_grid=dict(
        n_neighbors=[200]
    ),
    X=X_train,
    y=Y_train
)

gbc_best_params, gbc_best_score = GridSearchCVWrapper(
    model=GradientBoostingClassifier(),
    param_grid=dict(
        loss=["exponential"],
        learning_rate=[0.1],
        n_estimators=[100],
        max_depth=[3],
        min_samples_split=[2],
        min_samples_leaf=[1],
        max_features=["sqrt"]
    ),
    X=X_train,
    y=Y_train
)

# print(test_Y_train)
# np.save("X_test.npy",X_test)
# np.save("Y_test.npy",test_Y_train)
# exit(0)

rf_clf = RandomForestClassifier(**rf_best_params).fit(X_train, Y_train)
knn_clf = KNeighborsClassifier(**knn_best_params).fit(X_train, Y_train)
lg_clf = LogisticRegression(**lg_best_params).fit(X_train, Y_train)
svc_clf = SVC(**svc_best_params).fit(X_train, Y_train)
gbc = GradientBoostingClassifier(**gbc_best_params)
gbc_clf = gbc.fit(X_train, Y_train)

gbc_clf = gbc.fit(X_train, Y_train)

rf_proba  = rf_clf.predict_proba(X_test)
knn_proba = knn_clf.predict_proba(X_test)
lg_proba  = lg_clf.predict_proba(X_test)
svc_proba = svc_clf.predict_proba(X_test)
gbc_proba = gbc_clf.predict_proba(X_test)

# integrate_proba = (rf_proba + knn_proba + lg_proba + svc_proba + gbc_proba)/5
integrate_proba = (knn_proba + lg_proba + svc_proba + gbc_proba)/4
# Create Ensemble
df_preds = pd.DataFrame({
        'KNeighborsClassifier': knn_proba[:,1],
        'RandomForestClassifier': rf_proba[:,1],
        'GradientBoostingClassifier': gbc_proba[:,1],
        'SVC': svc_proba[:,1],
        'Integrate':integrate_proba[:,1]
    })

#df_preds.loc[:,'Ensemble'] = df_preds.mean(axis=1)
for col in df_preds:
    print(col)
    threshold = 0.5
    print(metrics.classification_report(Y_test, np.where(1 - df_preds.loc[:,col] > threshold, 0, 1)))
    
probas = dict(
    GradientBoostingClassifier=gbc_proba,
    KNeighborsClassifier=knn_proba,
    LogisticRegression=lg_proba,
    RandomForestClassifier=rf_proba,
    SVC=svc_proba
)

# plot_multi_recall_vs_decision_boundary(probas, test_Y_train)
# precision_recall_thershold(probas, test_Y_train)
# plot_roc_curves(df_preds, test_Y_train)
# plot_recall_vs_decision_boundary()

df_preds.to_csv(f"preds{mode}.csv",header=False,index=False)
np.save(f"test_label{mode}.npy",Y_test)
# pd.DataFrame(X_train).to_csv("X_train.csv")
# pd.DataFrame(Y_train).to_csv("Y_train.csv")
# pd.DataFrame(X_test).to_csv("X_test.csv")
# pd.DataFrame(Y_test).to_csv("Y_test.csv")

tok = time.time()
print("Time:", tok-tik)
