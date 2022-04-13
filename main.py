import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random
import librosa, IPython
import librosa.display as lplt

df = pd.read_csv('./data/features_3_sec.csv')
df = df.drop(labels="filename", axis=1)
# df.filename.unique()

# foo = df.iloc[:, :-1]
# bar = np.array(foo)

class_list = df.iloc[:, -1]
converter = LabelEncoder()

y = converter.fit_transform(class_list)

fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:, :-1], dtype=float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# X_train, X_test = train_test_split(X, test_size=0.33)
# y_train, y_test = train_test_split(y, test_size=0.33)

print(X_train)
print("==============")
print(X_test)
print("==============")
print(y_train)
print("==============")
print(y_test)

print("sadasd")
