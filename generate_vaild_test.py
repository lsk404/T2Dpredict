import numpy as np
import pandas as pd
import os
import smote
import random
df = pd.read_excel("C:/Users/lsk/Desktop/CG2405code/data/test/ground_true.xlsx")
arr = df.values
label0 = []
label1 = []
label2 = []

ecgs = os.listdir("C:/Users/lsk/Desktop/CG2405code/data/test/npy/")
ecgs = [ecgs[i][:-4] for i in range(len(ecgs))]
for i in range(len(df)):
    if(str(df.iloc[i,0]) in ecgs):
        if(df.iloc[i,2] == 1):
            if(df.iloc[i,3] == 1):
                label2.append(df.iloc[i,0])
            else:
                label1.append(df.iloc[i,0])
        else:
            label0.append(df.iloc[i,0])
print(len(label0))
print(len(label1))
print(len(label2))
label0 = np.random.choice(label0,300)
## save label0
t = 1
random.shuffle(label0)
random.shuffle(label1)
random.shuffle(label2)
for i in label0[:len(label0)//2]:
    np.save(f"data/test/label0_{t}.npy",np.load("C:/Users/lsk/Desktop/CG2405code/data/test/npy/"+str(i)+".npy"))
    t += 1
for i in label0[len(label0)//2:]:
    np.save(f"data/valid/label0_{t}.npy",np.load("C:/Users/lsk/Desktop/CG2405code/data/test/npy/"+str(i)+".npy"))
    t += 1

# save label1
t = 1
for i in label1[:len(label1)//2]:
    np.save(f"data/test/label1_{t}.npy",np.load("C:/Users/lsk/Desktop/CG2405code/data/test/npy/"+str(i)+".npy"))
    t += 1
for i in label1[len(label1)//2:]:
    np.save(f"data/valid/label1_{t}.npy",np.load("C:/Users/lsk/Desktop/CG2405code/data/test/npy/"+str(i)+".npy"))
    t += 1

# save label2
arrlst = []
for x in label2[:len(label2)//2]:
    arr = np.load("C:/Users/lsk/Desktop/CG2405code/data/test/npy/"+str(x)+".npy")
    arrlst.append(arr.reshape(-1))
print(len(arrlst))
arr_test = np.array(arrlst)
print(arr_test.shape)
Smote_test = smote.Smote(arr_test,N=400)
dataset = Smote_test.over_sampling()
t = 1
for data in dataset:
    np.save(f"data/test/label2_{t}.npy",data.reshape(12,-1))
    t+=1

arrlst.clear();t=1
for x in label2[len(label2)//2:]:
    arr = np.load("C:/Users/lsk/Desktop/CG2405code/data/test/npy/"+str(x)+".npy")
    arrlst.append(arr.reshape(-1))
print(len(arrlst))
arr_test = np.array(arrlst)
print(arr_test.shape)
Smote_test = smote.Smote(arr_test,N=400)
dataset = Smote_test.over_sampling()
t = 1
for data in dataset:
    np.save(f"data/valid/label2_{t}.npy",data.reshape(12,-1))
    t+=1