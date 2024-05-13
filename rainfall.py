#importin the modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


df=pd.read_csv('weather.csv')
#performing the recq changes like deleting and filling the empty columns
df.drop(['WindDir9am','WindDir3pm'],axis=1,inplace=True)
df.Sunshine.fillna(df.Sunshine.mean(),inplace=True)
df.WindGustSpeed.fillna(df.WindGustSpeed.mean(),inplace=True)
df.drop(['WindSpeed9am','WindGustDir'],axis=1,inplace=True)
df['RainTomorrow']=df['RainTomorrow'].map({'Yes':1,'No':0})
df['RainToday']=df['RainToday'].map({'Yes':1,'No':0})
df.isna().sum()

df=pd.get_dummies(df)

df.columns

#assigning the new variable x and y
y=df['RainTomorrow']
x=df[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustSpeed', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RISK_MM']]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2529)

model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#to get your own result please change the integer values inside the square bracket
if model.predict([[20,30,0.1,3.2,6.0,29,19,69,30,1010.5,1011,5,4,15,25,0,3.2]])=='1':
  print("\n Rain Tmrw \n")
else:
  print("\n No Rain tmrw \n")
print("accuracy",accuracy_score(y_test,y_pred)*100,"%")
