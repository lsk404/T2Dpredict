import numpy as np
import pandas as pd
importance = np.load('feature_importance_.npy')

importance.reshape(1,-1)
importance = importance[:478]
pd.DataFrame(importance).to_csv("importance.csv",header=None,index=None)
# sort
sorted_idx = np.argsort(importance)[::-1]
importance = importance[sorted_idx]
print(importance)
print(sorted_idx)
