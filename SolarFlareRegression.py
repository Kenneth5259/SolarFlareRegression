# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('flare.data2')
X = dataset.iloc[:, :-3].values
y_C = dataset.iloc[:, 10].values
y_M = dataset.iloc[:, 11].values
y_X = dataset.iloc[:, 12].values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

classTransformer = ColumnTransformer([("Class", OneHotEncoder(), [0])], remainder='passthrough')
X = classTransformer.fit_transform(X)
spotTransformer = ColumnTransformer([("Spot", OneHotEncoder(), [6])], remainder='passthrough')
X = spotTransformer.fit_transform(X)
spotDistTransformer = ColumnTransformer([("Spot", OneHotEncoder(), [12])], remainder='passthrough')
X = spotDistTransformer.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_trainC, X_testC, y_trainC, y_testC = train_test_split(X, y_C, test_size = 0.2, random_state = 0)
X_trainM, X_testM, y_trainM, y_testM = train_test_split(X, y_M, test_size = 0.2, random_state = 0)
X_trainX, X_testX, y_trainX, y_testX = train_test_split(X, y_X, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressorC = RandomForestRegressor(n_estimators=1000)
regressorC = regressorC.fit(X_trainC, y_trainC)
y_predC = np.around(np.array(regressorC.predict(X_testC)))
y_predC = y_predC.astype(int)

regressorM = RandomForestRegressor(n_estimators=1000)
regressorM = regressorM.fit(X_trainM, y_trainM)
y_predM = np.around(np.array(regressorM.predict(X_testM)))
y_predM = y_predM.astype(int)

regressorX = RandomForestRegressor(n_estimators=1000)
regressorX = regressorX.fit(X_trainX, y_trainX)
y_predX = np.around(np.array(regressorX.predict(X_testX)))
y_predX = y_predX.astype(int)

from sklearn.metrics import classification_report

reportC =np.vstack((y_testC, y_predC)).T
reportC = pd.DataFrame (reportC, columns=['Expected', 'Predicted'])
reportM =np.vstack((y_testM, y_predM)).T
reportM = pd.DataFrame (reportM, columns=['Expected', 'Predicted'])
reportX =np.vstack((y_testX, y_predX)).T
reportX = pd.DataFrame (reportX, columns=['Expected', 'Predicted'])
print(classification_report(reportC['Expected'], reportC['Predicted']))
print(classification_report(reportM['Expected'], reportC['Predicted']))
print(classification_report(reportX['Expected'], reportC['Predicted']))


#print(y_predC)
#print(y_testC)
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

#print(y_testC)
#print(y_testM)
#print(y_testX)