import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("house_data.csv")

X = data[['area', 'bedrooms']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

area = int(input("Enter Area: "))
bedrooms = int(input("Enter Bedrooms: "))

prediction = model.predict([[area, bedrooms]])

print("Predicted Price:", prediction[0])