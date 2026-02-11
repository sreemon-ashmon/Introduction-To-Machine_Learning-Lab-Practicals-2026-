import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

file_path = "position_salaries.csv"
df = pd.read_csv(file_path)

X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

degree = 4
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

plt.scatter(X, y)
plt.plot(X, model.predict(X_poly))
plt.title("Polynomial Regression (Degree 4)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.show()
