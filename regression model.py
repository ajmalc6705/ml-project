import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

#import dataset
data = pd.read_csv("auto-mpg.csv")
data['horsepower'] = data['horsepower'].replace('?','100')
data.horsepower.mean()
data.mpg.mean()
data.isnull().sum()

len(data['car name'].value_counts())
data['car name'] = data['car name'].str.split(' ').str.get(0)

data['car name'] = data['car name'].replace(['chevroelt','chevy'],'chevrolet')
data['car name'] = data['car name'].replace(['vokswagen','vw'],'volkswagen')
data['car name'] = data['car name'].replace('maxda','mazda')
data['car name'] = data['car name'].replace('toyouta','toyota')
data['car name'] = data['car name'].replace('mercedes','mercedes-benz')
data['car name'] = data['car name'].replace('nissan','datsun')
data['car name'] = data['car name'].replace('capri','ford')
data['car name'].value_counts()
data['car name'].isnull().sum()

x = data.iloc[:,[1,2,3,4,5,6,7]].values
y = data.iloc[:,0].values

#Categorical Variable

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
x[:,7] = lb.fit_transform(x[:,7])

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(categorical_features = [7])
x = onehot.fit_transform(x).toarray()


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)


#Splitting data set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 0)

# multiple linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)

lr.score(xtrain,ytrain)

from sklearn.metrics import r2_score
print("Accuracy of the linear model is:",r2_score(ytest,ypred)*100)

#Polynomial regression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtrain,ytrain)

#add polynomial features to model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
xtrainpoly = poly_reg.fit_transform(xtrain)
xtestpoly = poly_reg.fit_transform(xtest)


poly_reg.fit(xtrainpoly,ytrain)
lr2 = LinearRegression()
lr2.fit(xtrainpoly,ytrain)
ypred_poly = lr2.predict(xtestpoly)

r2_score(ytest,ypred_poly)

#Decision tree Regression
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 13)
dtr.fit(xtrain,ytrain)

ypred_dtr = dtr.predict(xtest) 
r2_score(ytest,ypred_dtr)

#Random forest
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 10,random_state = 0)
rfr.fit(xtrain,ytrain)

ypred_rfr = rfr.predict(xtest)
r2_score(ytest,ypred_rfr)

#Support Vector Regression
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(xtrain,ytrain)

ypred_svr = svr.predict(xtest)
r2_score(ytest,ypred_svr)








