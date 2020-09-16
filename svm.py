# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
data = pd.read_csv("data.csv")

# %%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

# %%
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
# scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha= 0.2)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha= 0.2)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

# %%
# normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=7)
#%% SVM
    
from sklearn.svm import SVC
svc=SVC(kernel='poly')
model_poly=svc.fit(x_train, y_train)

y_prediction=model_poly.predict(x_test)

#Başarı oranı
from sklearn.metrics import accuracy_score
accuracy_poly = accuracy_score(y_test, y_prediction)
print('Accuracy_Poly:', accuracy_poly)

from sklearn.svm import SVC
svc=SVC(kernel='linear')
model_linear=svc.fit(x_train, y_train)

y_prediction=model_linear.predict(x_test)

#Başarı oranı
from sklearn.metrics import accuracy_score
accuracy_linear = accuracy_score(y_test, y_prediction)
print('Accuracy_Linear:', accuracy_linear)

from sklearn.svm import SVC
svc=SVC(kernel='rbf')
model_rbf=svc.fit(x_train, y_train)

y_prediction=model_rbf.predict(x_test)

#Başarı oranı
from sklearn.metrics import accuracy_score
accuracy_rbf = accuracy_score(y_test, y_prediction)
print('Accuracy_Rbf:', accuracy_rbf)
#%% test


