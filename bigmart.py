import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis,skew
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

d1=pd.read_csv('Train.csv')
data=d1
print( d1.shape )
print()

print( data['Item_Fat_Content'].value_counts() )
print()
data['Item_Fat_Content']=data['Item_Fat_Content'].map({'LF':0,'Low Fat':0,'Regular':1,'low fat':0,'reg':1})

print( data.isnull().sum() )
print()

yy=data.groupby('Item_Identifier')['Item_Weight'].mean()
yy=dict(yy)
data['Item_Weight'].fillna(-1.0,inplace=True)
def ff(xx,d):
    if(d==-1.0):
        return yy[xx]
    else:
        return d
data['Item_Weight']=data.apply(lambda x : ff(x['Item_Identifier'],x['Item_Weight']),axis=1)

print( data.isnull().sum() )
print()

print( data.groupby('Item_Type')['Item_Weight'].median() )
print()
data['Item_Weight'].fillna(10.0,inplace=True)

print( data.groupby('Outlet_Type')['Outlet_Size'].value_counts() )
print()
print( data.groupby(['Outlet_Type','Outlet_Size'])['Item_Outlet_Sales'].mean() )
print()
data['Outlet_Size'].fillna('Small',inplace=True)

data['Item_MRP'].hist(bins=200,figsize=(10,6))
plt.show()

print( np.where(data.Item_Visibility<=0) )
print()
yy=data[data.Item_Visibility>0.0].groupby('Item_Identifier')['Item_Visibility'].mean()
yy=dict(yy)
def ff(x,y):
    if(x==0.0):
        return yy[y]
    else:
        return x
data['Item_Visibility']=data.apply(lambda x : ff(x['Item_Visibility'],x['Item_Identifier']),axis=1)

ya=data.groupby('Item_Identifier')['Item_Visibility'].mean()
ya=dict(ya)
def ff(x,y):
    return (x/ya[y])
data['itemimportance']=data.apply(lambda x : ff(x['Item_Visibility'],x['Item_Identifier']),axis=1)
print( data['itemimportance'].describe() )
print()

data['years']=2018-data['Outlet_Establishment_Year']
del data['Outlet_Establishment_Year']

lb=LabelEncoder()
lb.fit(data['Item_Type'])
data['Item_Type']=lb.transform(data['Item_Type'])

lb=LabelEncoder()
lb.fit(data['Outlet_Identifier'])
data['Outlet_Identifier']=lb.transform(data['Outlet_Identifier'])

lb=LabelEncoder()
lb.fit(data['Outlet_Size'])
data['Outlet_Size']=lb.transform(data['Outlet_Size'])

lb=LabelEncoder()
lb.fit(data['Outlet_Location_Type'])
data['Outlet_Location_Type']=lb.transform(data['Outlet_Location_Type'])

lb=LabelEncoder()
lb.fit(data['Outlet_Type'])
data['Outlet_Type']=lb.transform(data['Outlet_Type'])

print( data.isnull().sum() )
print()
print( data.describe() )
print()

data.boxplot(by='Outlet_Type',column='Item_Outlet_Sales')
plt.show()

del data['Item_Identifier']
test=data[8500:]
data=data[0:8500]
yd=data['Item_Outlet_Sales']
del data['Item_Outlet_Sales']
del test['Item_Outlet_Sales']
print( data.shape,yd.shape,test.shape )
print()
yy=yd

print( np.where(np.isnan(data)) )
print()
print( data.isnull().sum() )
print()

print("Performing Simple Linear Regression:")
avg=0.0
skf=KFold(n_splits=5)
skf.get_n_splits(data)
for ti,tj in skf.split(data):
    dx,tx=data.iloc[ti],data.iloc[tj]
    dy,ty=yy[ti],yy[tj]
    lm=make_pipeline(MinMaxScaler(), linear_model.LinearRegression(n_jobs=-1))
    lm.fit(dx,dy)
    yu=np.sqrt(mean_squared_error(y_true=ty,y_pred=lm.predict(tx)))
    avg=avg+yu
    print(yu)
print("AVG  RMSE::",avg/5)
print()

print("Performing Simple Elasticnet Regression:")
avg=0.0
skf=KFold(n_splits=5)
skf.get_n_splits(data)
for ti,tj in skf.split(data):
    dx,tx=data.iloc[ti],data.iloc[tj]
    dy,ty=yy[ti],yy[tj]
    lm=make_pipeline(StandardScaler(), linear_model.ElasticNet(l1_ratio=0.6,alpha=0.001))
    lm.fit(dx,dy)
    yu=np.sqrt(mean_squared_error(y_true=ty,y_pred=lm.predict(tx)))
    avg=avg+yu
    print(yu)
print("AVG  RMSE::",avg/5)
print()

print("Performing Decision Tree Regression:")
avg=0.0
skf=KFold(n_splits=5)
skf.get_n_splits(data)
for ti,tj in skf.split(data):
    dx,tx=data.iloc[ti],data.iloc[tj]
    dy,ty=yy[ti],yy[tj]
    lm=DecisionTreeRegressor(max_depth=5)
    lm.fit(dx,dy)
    yu=np.sqrt(mean_squared_error(y_true=ty,y_pred=lm.predict(tx)))
    avg=avg+yu
    print(yu)
print("AVG  RMSE::",avg/5)
print()

print("Performing Random Forest Tree Regression:")
avg=0.0
skf=KFold(n_splits=5)
skf.get_n_splits(data)
for ti,tj in skf.split(data):
    dx,tx=data.iloc[ti],data.iloc[tj]
    dy,ty=yy[ti],yy[tj]
    lm=RandomForestRegressor(max_depth=5,n_jobs=-1,n_estimators=100)
    lm.fit(dx,dy)
    yu=np.sqrt(mean_squared_error(y_true=ty,y_pred=lm.predict(tx)))
    avg=avg+yu
    print(yu)
print("AVG  RMSE::",avg/5)
print()
