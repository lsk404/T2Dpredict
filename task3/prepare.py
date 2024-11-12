import xml.sax
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET

base_train = np.load("data/basedata/train.npy")
train_ecgs = os.listdir("C:/Users/lsk/Desktop/CG2405code/data/train/ecg")
train_ground_true = pd.read_excel("C:/Users/lsk/Desktop/CG2405code/data/train/ground_true.xlsx")
begin_df = pd.read_excel("C:/Users/lsk/Desktop/CG2405code/data/train/Baseline_characteristics.xlsx")
train_beg = begin_df.iloc[2400:,2].values
train_end = train_ground_true.iloc[2400:,1].values
for i in range(len(train_beg)):
    train_beg[i] = int(train_beg[i])
for i in range(len(train_end)):
    train_end[i] = int(train_end[i].split('-')[0])
train_label = np.array(train_end - train_beg)

train_ecgs = [int(i.split('_')[0]) for i in train_ecgs]

save_train_lst = []
# print(train_label)

X_1_lst = []
X_2_lst = []
Y_lst = []

useful = ["VentricularRate","PQInterval","PDuration","QRSDuration","QTInterval","QTCInterval","RRInterval","PPInterval","SokolovLVHIndex","PAxis","RAxis","TAxis","QTDispersion","QTDispersionBazett","POnset","POffset","QOnset","QOffset","TOffset"]
for i in range(2400,len(train_ground_true)):
    if train_ground_true.iloc[i,0] in train_ecgs:
        id = train_ground_true.iloc[i,0]
        Path = "C:/Users/lsk/Desktop/CG2405code/data/train/ecg/"+str(id)+"_20205_2_0.xml"
        tree = ET.parse(Path)
        root = tree.getroot()
        tg = root.find("RestingECGMeasurements")
        x1 = base_train[i,:-2].reshape(1,-1)
        X_1_lst.append(x1)
        # print(i,id,Path)
        x2 = []
        for x in useful:
            if (tg.find(x) == None or tg.find(x).text == None):
                x2.append(np.nan)
            else:
                x2.append(tg.find(x).text)
        X_2_lst.append(x2)
        y = np.array([train_label[i-2400]]).reshape(1,-1)
        Y_lst.append(y)

X_1 = np.vstack(X_1_lst)
X_2 = np.vstack(X_2_lst)
Y = np.vstack(Y_lst)

X_2_df = pd.DataFrame(X_2)
X_2_df = X_2_df.astype(float)
X_2_df.fillna(X_2_df.mean(),inplace=True)
X_2_df = X_2_df.dropna(axis=1)
X_2 = X_2_df.values
print(X_1.shape)
print(X_2_df.shape)
print(Y.shape)
save_train = np.hstack((X_1,X_2_df,Y))
np.save("data/train.npy",save_train)
# print(save_train.shape)
