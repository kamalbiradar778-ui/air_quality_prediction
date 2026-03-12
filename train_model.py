import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("data/air_quality.csv")  # Make sure this path exists

# Features and target
X = data[['PM2.5','PM10','NO2','SO2','CO']]
y = data['AQI']

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
pickle.dump(model, open("aqi_model.pkl","wb"))

print("✅ Model trained and saved as aqi_model.pkl")
