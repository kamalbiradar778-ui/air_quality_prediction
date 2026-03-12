import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("data/air_quality.csv")

X = data[['PM2.5','PM10','NO2','SO2','CO']]
y = data['AQI']

model = LinearRegression()
model.fit(X,y)

pickle.dump(model,open("aqi_model.pkl","wb"))

print("Model trained successfully")