import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample data: square footage, number of bedrooms, number of bathrooms, and price
data = {
    'square_footage': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'bedrooms': [3, 3, 3, 4, 4, 4, 5, 5, 5, 5],
    'bathrooms': [2, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    'price': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['square_footage', 'bedrooms', 'bathrooms']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plot the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

# Predict the price of a new house
new_house = np.array([[2200, 4, 3]])
predicted_price = model.predict(new_house)
print(f"The predicted price for the house is: ${predicted_price[0]:.2f}")
