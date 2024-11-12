import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pred = np.load("preds.npy")

label = np.load("data/test.npy")[:,-1]

plt.scatter(label,pred,s=10,c='red',marker='o')
mx = max(max(label),max(pred))
plt.xlim(40,mx)
plt.ylim(40,mx)
plt.xlabel("True value")
plt.ylabel("pred value")
plt.plot([0,mx],[0,mx],color="blue")
plt.title("Comparison of Predictions with Values")
plt.legend()
plt.savefig("pred_vs_true.png")
# plt.show()
