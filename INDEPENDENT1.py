
import pandas as pd
data=pd.read_csv("50_Startups-1.csv")
print(data.isna().sum())
from sklearn.impute import SimpleImputer
si=SimpleImputer()
data[['R&D Spend','Administration','Marketing Spend']]=si.fit_transform(data[['R&D Spend','Administration','Marketing Spend']])
print(data.isna().sum())
si=SimpleImputer(strategy="most_frequent")
data[['State']]=si.fit_transform(data[['State']])
print(data.isna().sum())

c=data.corr()['Profit']
print(c)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["State"]=le.fit_transform(data['State'])
c=data.corr()['Profit']
print(c)
X=data.iloc[:,0].values
Y=data.iloc[:,-1].values
X=X.reshape(X.shape[0],1)
Y=Y.reshape(Y.shape[0],1)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit_transform(X)

from sklearn.model_selection import train_test_split  
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=2)

import matplotlib.pyplot as plt
plt.plot(data['Profit'],data['R&D Spend'],label="R&D Spend")
plt.xlabel("Spend")
plt.ylabel("Profit")
plt.legend()
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_trans=poly.fit_transform(X_train)

from sklearn.linear_model import LinearRegression
Lr=LinearRegression()
Lr.fit(x_trans,Y_train)
x_trans_test=poly.fit_transform(X_test)
Y_prepoly=Lr.predict(x_trans_test)

from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(random_state=0,max_depth=5)
reg.fit(X,Y)
Y_predtr=reg.predict(X_test)

from sklearn.svm import SVR
svr1=SVR(kernel="poly",degree=5)
svr1.fit(X,Y)
Y_presvrp=svr1.predict(X_test)

from sklearn.neighbors import KNeighborsRegressor
knnr=KNeighborsRegressor(n_neighbors=3)
knnr.fit(X,Y)
Y_pre_knn=knnr.predict(X_test)

from sklearn.metrics import r2_score
scorepoly=r2_score(Y_test,Y_prepoly)
print('PR',scorepoly)
scoredtr=r2_score(Y_test,Y_predtr)
print("DTR",scoredtr)
scoresvrp=r2_score(Y_test,Y_presvrp)
print('SVR',scoresvrp)
scoreKNN=r2_score(Y_test,Y_pre_knn)
print("KNNR",scoreKNN)
