from imblearn.under_sampling import NearMiss
import pickle as pkl
import pandas as pd 

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')

us = NearMiss( n_neighbors=3, version=2,)
X_train_res, y_train_res = us.fit_resample(X_train, y_train)

