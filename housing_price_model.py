import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("housing.csv")
data.dropna(inplace=True)

from sklearn.model_selection import train_test_split
X=data.drop(['median_house_value'],axis=1)
Y=data['median_house_value']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
train_data= X_train.join(Y_train)
train_data['total_rooms']=np.log(train_data['total_rooms']+1)
train_data['total_bedrooms']=np.log(train_data['total_bedrooms']+1)
train_data['population']=np.log(train_data['population']+1)
train_data['households']=np.log(train_data['households']+1)

train_data=train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)

train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_ratio'] = train_data['total_rooms'] / train_data['households']

from sklearn.linear_model import LinearRegression

X_train, Y_train= train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']
reg=LinearRegression()
reg.fit(X_train,Y_train)

test_data=X_test.join(Y_test)
test_data['total_rooms']=np.log(test_data['total_rooms']+1)
test_data['total_bedrooms']=np.log(test_data['total_bedrooms']+1)
test_data['population']=np.log(test_data['population']+1)
test_data['households']=np.log(test_data['households']+1)

test_data=test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)
test_data['bedroom_ratio']=test_data['total_bedrooms']/test_data['total_rooms']
test_data['household_ratio']=test_data['total_rooms']/test_data['households']

test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

X_test, Y_test= test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor()
forest.fit(X_train,Y_train)
print(forest.score(X_test,Y_test))
import joblib
joblib.dump(model, "house_model.pkl", compress=3)

import pickle
with open("house_model.pkl","wb") as f:
  pickle.dump(forest,f)

print("Model saved successfully!!")


