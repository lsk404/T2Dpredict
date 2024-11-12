from sklearn.preprocessing import MinMaxScaler
from utils.utils import (GridSearchCVWrapper, 
                          one_hot_encode, 
                          precision_recall_thershold, 
                          plot_recall_vs_decision_boundary, 
                          plot_multi_recall_vs_decision_boundary,
                          plot_roc_curves,
                          plot_bootstrap_roc,
                          bootstrap_model, 
                          roc_interp)
import pandas as pd
import numpy as np

ground_true_path_train = "./data/train/ground_true.xlsx"
Baseline_characteristics_path_train = "./data/train/Baseline_characteristics.xlsx"
life_style_path_train = "./data/train/life_style.xlsx"
NMR_path_train = "./data/train/NMR.xlsx"
# 提取数据
id = pd.read_excel(ground_true_path_train).iloc[:,0]
train_label = pd.read_excel(ground_true_path_train).iloc[:,2:4]
Baseline_characteristics = pd.read_excel(Baseline_characteristics_path_train).iloc[:,1:]
Baseline_characteristics.fillna(Baseline_characteristics.median(), inplace=True)
life_style = pd.read_excel(life_style_path_train).iloc[:,1:]

NMR = pd.read_excel(NMR_path_train).iloc[:,1:] # 读取表格文件
NMR_0 = NMR.iloc[0:2400,:]
NMR_0.fillna(NMR_0.median(), inplace=True)
NMR_1 = NMR.iloc[2400:,:]
NMR_1.fillna(NMR_1.median(), inplace=True)
NMR = pd.concat([NMR_0,NMR_1], axis=0)

life_style["f.874.0.0"] = life_style["f.874.0.0"].replace(-1,0) # 走路的时间
life_style["f.914.0.0"] = life_style["f.914.0.0"].replace(-1,0) # 剧烈运动的时间
life_style["f.20077.0.0"].fillna(0,inplace=True)

indices = life_style[life_style["f.20160.0.0"] == 1].index
for col in ["f.2644.0.0","f.2867.0.0","f.2887.0.0","f.2907.0.0","f.2926.0.0","f.3436.0.0","f.3456.0.0","f.3466.0.0"]:
    median = life_style.loc[indices,col].median()
    life_style.loc[life_style.index.isin(indices) & life_style[col].isna(),col] = median

indices = life_style[life_style["f.20160.0.0"] == 0].index
for col in ["f.2644.0.0","f.2867.0.0","f.2887.0.0","f.2907.0.0","f.2926.0.0","f.3436.0.0","f.3456.0.0","f.3466.0.0"]:
    median = life_style.loc[indices,col].median()
    life_style.loc[life_style.index.isin(indices) & life_style[col].isna(),col] = median

life_style[:2400].fillna(life_style[:2400].median(),inplace=True)
life_style[2400:].fillna(life_style[2400:].median(),inplace=True)

df = pd.concat([Baseline_characteristics,life_style,NMR],axis=1)

Data_Dictionary = pd.read_excel("./data/Data_Dictionary.xlsx")

categorical_cols = Data_Dictionary[Data_Dictionary["ValueType"]=="Categorical single"].iloc[:,0]
continuous_cols = Data_Dictionary[Data_Dictionary["ValueType"]!="Categorical single"].iloc[:,0]
print("1",categorical_cols.shape)
print("2",continuous_cols.shape)
X_train_categorical = pd.concat(map(lambda col : one_hot_encode(df[col],col), categorical_cols),axis=1)
X_train = pd.concat([X_train_categorical, df[continuous_cols]], axis = 1)

ground_true_path_test = "./data/test/ground_true.xlsx"
Baseline_characteristics_path_test = "./data/test/Baseline_characteristics.xlsx"
life_style_path_test = "./data/test/life_style.xlsx"
NMR_path_test = "./data/test/NMR.xlsx"

# 提取数据
id = pd.read_excel(ground_true_path_test).iloc[:,0]
test_label = pd.read_excel(ground_true_path_test).iloc[:,2:4]
Baseline_characteristics = pd.read_excel(Baseline_characteristics_path_test).iloc[:,1:]
Baseline_characteristics.fillna(Baseline_characteristics.median(), inplace=True)
life_style = pd.read_excel(life_style_path_test).iloc[:,1:]
NMR = pd.read_excel(NMR_path_test).iloc[:,1:] # 读取表格文件
NMR_0 = NMR.iloc[0:600,:]
NMR_0.fillna(NMR_0.median(), inplace=True)
NMR_1 = NMR.iloc[600:,:]
NMR_1.fillna(NMR_1.median(), inplace=True)
NMR = pd.concat([NMR_0,NMR_1], axis=0)

life_style["f.874.0.0"] = life_style["f.874.0.0"].replace(-1,0) # 走路的时间
life_style["f.914.0.0"] = life_style["f.914.0.0"].replace(-1,0) # 剧烈运动的时间
life_style["f.20077.0.0"].fillna(0,inplace=True)

indices = life_style[life_style["f.20160.0.0"] == 1].index
for col in ["f.2644.0.0","f.2867.0.0","f.2887.0.0","f.2907.0.0","f.2926.0.0","f.3436.0.0","f.3456.0.0","f.3466.0.0"]:
    median = life_style.loc[indices,col].median()
    life_style.loc[life_style.index.isin(indices) & life_style[col].isna(),col] = median

indices = life_style[life_style["f.20160.0.0"] == 0].index
for col in ["f.2644.0.0","f.2867.0.0","f.2887.0.0","f.2907.0.0","f.2926.0.0","f.3436.0.0","f.3456.0.0","f.3466.0.0"]:
    median = life_style.loc[indices,col].median()
    life_style.loc[life_style.index.isin(indices) & life_style[col].isna(),col] = median

life_style[:600].fillna(life_style[:600].median(),inplace=True)
life_style[600:].fillna(life_style[600:].median(),inplace=True)
df = pd.concat([Baseline_characteristics,life_style,NMR],axis=1)

X_test_categorical = pd.concat(map(lambda col : one_hot_encode(df[col],col), categorical_cols),axis=1)
X_test = pd.concat([X_test_categorical, df[continuous_cols]], axis = 1)

for missing_feature in X_train.columns:
    if missing_feature not in X_test.columns:
        X_test[missing_feature] = 0
for missing_feature in X_test.columns:
    if missing_feature not in X_train.columns:
        X_train[missing_feature] = 0

X_test = X_test.reindex(columns=X_train.columns.tolist())


scale = MinMaxScaler(feature_range=(-1, 1)).fit(X_train[continuous_cols])
train_cols = X_train.columns
categorical_cols = [c for c in train_cols if (c not in continuous_cols.values)]
X_train_scale = np.hstack((scale.transform(X_train[continuous_cols]), X_train[categorical_cols]))
# pd.DataFrame(X_train_scale).to_csv("X_train.csv")

scale = MinMaxScaler(feature_range=(-1, 1)).fit(X_test[continuous_cols])
test_cols = X_test.columns
categorical_cols = [c for c in test_cols if (c not in continuous_cols.values)]
X_test_scale = np.hstack((scale.transform(X_test[continuous_cols]), X_test[categorical_cols]))

# np.save("X_train.npy",X_train_scale)
# np.save("Y_train.npy",label)
# np.save("X_test.npy",X_test_scale)
# np.save("Y_test.npy",test_label)

train = np.hstack((X_train_scale, train_label.values))
test = np.hstack((X_test_scale, test_label.values))

print(train.shape)
print(test.shape)
np.save("train.npy",train)
np.save("test.npy",test)


col = np.hstack((continuous_cols.values.reshape(1,-1),np.array(categorical_cols).reshape(1,-1)))
print(col)
print(len(col[0]))
pd.DataFrame(col.reshape(-1,1)).to_csv("header.csv",header=None,index=None)