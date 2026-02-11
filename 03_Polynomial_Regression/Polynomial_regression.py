import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "position_salaries.csv"
df = pd.read_csv(file_path)

X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

def viz_polynomial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Polynomial Regression')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()
    return

viz_polynomial()
