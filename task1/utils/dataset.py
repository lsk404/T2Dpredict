from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class dataSet_loader(Dataset):
    def __init__(self):
        ground_true_path = "C:/Users/lsk/Desktop/CG2405code/data/train/ground_true.xlsx"
        Baseline_characteristics_path = "C:/Users/lsk/Desktop/CG2405code/data/train/Baseline_characteristics.xlsx"
        life_style_path = "C:/Users/lsk/Desktop/CG2405code/data/train/life_style.xlsx"
        NMR_path = "C:/Users/lsk/Desktop/CG2405code/data/train/NMR.xlsx"
        id = pd.read_excel(ground_true_path).iloc[:,0]
        value = pd.read_excel(ground_true_path).iloc[:,2]
        Baseline_characteristics = pd.read_excel(Baseline_characteristics_path).iloc[:,1:]
        life_style = pd.read_excel(life_style_path).iloc[:,1:]
        NMR = pd.read_excel(NMR_path).iloc[:,1:] # 读取表格文件
        self.id = id
        self.X = pd.concat([Baseline_characteristics,life_style,NMR],axis=1)
        self.val = value
    def __getitem__(self, index):
        return self.id[index],self.X[index],self.val[index]
    
    