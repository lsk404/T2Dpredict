import xml.sax
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET

base_test = np.load("data/basedata/test.npy")
test_ecgs = os.listdir("C:/Users/lsk/Desktop/CG2405code/data/test/ecg")
test_ground_true = pd.read_excel("C:/Users/lsk/Desktop/CG2405code/data/test/ground_true.xlsx")
begin_df = pd.read_excel("C:/Users/lsk/Desktop/CG2405code/data/test/Baseline_characteristics.xlsx")
test_beg = begin_df.iloc[600:,2].values
test_end = test_ground_true.iloc[600:,1].values
for i in range(len(test_beg)):
    test_beg[i] = int(test_beg[i])
for i in range(len(test_end)):
    test_end[i] = int(test_end[i].split('-')[0])
test_label = np.array(test_end - test_beg)

test_ecgs = [int(i.split('_')[0]) for i in test_ecgs]

save_test_lst = []
print(test_label)

X_1_lst = []
X_2_lst = []
Y_lst = []

useful = ["VentricularRate","PQInterval","PDuration","QRSDuration","QTInterval","QTCInterval","RRInterval","PPInterval","SokolovLVHIndex","PAxis","RAxis","TAxis","QTDispersion","QTDispersionBazett","POnset","POffset","QOnset","QOffset","TOffset"]
for i in range(600,len(test_ground_true)):
    if test_ground_true.iloc[i,0] in test_ecgs:
        id = test_ground_true.iloc[i,0]
        Path = "C:/Users/lsk/Desktop/CG2405code/data/test/ecg/"+str(id)+"_20205_2_0.xml"
        tree = ET.parse(Path)
        root = tree.getroot()
        tg = root.find("RestingECGMeasurements")
        x1 = base_test[i,:-2].reshape(1,-1)
        X_1_lst.append(x1)
        # print(i,id,Path)
        x2 = []
        for x in useful:
            if (tg.find(x) == None or tg.find(x).text == None):
                x2.append(np.nan)
            else:
                x2.append(tg.find(x).text)
        X_2_lst.append(x2)
        y = np.array([test_label[i-600]]).reshape(1,-1)
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
save_test = np.hstack((X_1,X_2_df,Y))
np.save("data/test.npy",save_test[:len(save_test)//2])
np.save("data/valid.npy",save_test[len(save_test)//2:])
print(save_test.shape)
