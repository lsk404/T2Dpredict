import numpy as np
import os
npz_path = "C:/Users/lsk/Desktop/CG2405code/data/test/npz/"
npy_path = "C:/Users/lsk/Desktop/CG2405code/data/test/npy/"

npz = os.listdir(npz_path)

for file in npz:
    arr = np.load(npz_path+file)
    arr = arr["arr_0"]
    np.save(npy_path+file[:-4],arr)