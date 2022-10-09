import pandas as pd 
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
# from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('https://raw.githubusercontent.com/BautistaDavid/WebApp_EconomiaLaboral/main/data/data_model.csv')
df = df.drop(columns=['afro','gitano','indigena','palenquero'])

# defining X and y 
y = df['empleado']
X = df.drop(columns='empleado')

# standardizing
scaler = StandardScaler().fit(X[['edad','años_edc','estrato']])

scaler_pickle = open("model/scaler.pickle","wb")
pkl.dump(scaler,scaler_pickle) 
scaler_pickle.close()   

X_scal = X 
X_scal[['edad','años_edc','estrato']] = scaler.transform(X[['edad','años_edc','estrato']])

# split for train and test sample 
X_train, X_test, y_train, y_test  = train_test_split(X_scal,y,test_size=0.3,random_state=777)
X_test.to_csv('data/X_test.csv',index=False)
y_test.to_csv('data/y_test.csv',index=False)

X_train.to_csv('data/X_train.csv',index=False)
y_train.to_csv('data/y_train.csv',index=False)

# 
os_us = SMOTETomek()
X_train_res, y_train_res = os_us.fit_resample(X_train, y_train)



# Grid to improve hyperparameters 
# param_grid = {'criterion' : ["gini", "entropy"],
#               'max_depth':[10,15,20,25,None],
#               'n_estimators':[100,120,150]}

# clf = GridSearchCV(RandomForestClassifier(),param_grid)
# clf.fit(X_train,y_train)
# clf.best_params_

# Modelinig a RandomForest 
clf = RandomForestClassifier(criterion='entropy',max_depth=None,n_estimators=100,random_state=777)
clf.fit(X_train_res,y_train_res)

# Using pickle to save the models 

clf_pickle = open("model/clf_randomforest.pickle","wb")
pkl.dump(clf,clf_pickle) 
clf_pickle.close()   





